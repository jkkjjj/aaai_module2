import math
import re
from .openai_helpers import chat_completion_with_retries
from .cross_episode_memory import CrossEpisodeMemory
from . import dacs as dacs_mod
import os

class OurAgent:
    def __init__(self, args, guiding_prompt: str = None): 
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent
        self.game_history = [] # Used by evolutionary LLM
        self.args = args
        self.user_prompt_evo = ""
        
        # Each node is a dict with keys: prompt, state_extractor, game_history, score, parent_idx, children_idxs
        self.nodes = []
        
        self.code = """
def extract_state(game_history):
    return "Game in progress."
"""

        self.best_node_idx = None
        self.best_score = float("-inf")
        self.last_episode_score = None
        self.freeze_on_win = getattr(self.args, 'freeze_on_win', True)
        self.win_freeze_threshold = getattr(self.args, 'win_freeze_threshold', 0)
        self.force_best_after_drop = getattr(self.args, 'force_best_after_drop', True)
        self.drop_threshold = getattr(self.args, 'drop_threshold', 50)

        # Auto-detection and latch for freezing after a detected win
        self.auto_freeze_on_win = True
        self.is_frozen = False
        # Internal: whether we auto-set the freeze threshold
        self._auto_set_threshold = False
        # Track last episode victory result
        self.last_episode_was_victory = False
        # After unfreezing due to missed win, force evolution in next start
        self.just_unfroze_due_to_missed_win = False

        # Cross-episode memory toggle
        self.enable_cross_mem = getattr(self.args, 'enable_cross_mem', True)

        # Cross-episode memory
        if self.enable_cross_mem:
            # Build output dir same structure as evaluation does: output/<game>/<agent_type>/<model_slug>/<timestamp>
            # For cross-episode we use base path output/<game>/<agent_type>/<model_slug>
            game_dir = getattr(self.args, 'output_path', 'output')
            game_dir = os.path.join(game_dir, getattr(self.args, 'game_name', 'game'))
            model_slug = getattr(self.args, 'llm_model', 'model').replace('/', '_').replace('\\', '_')
            agent_type = getattr(self.args, 'agent_type', 'our')
            self.cross_mem_dir = os.path.join(game_dir, agent_type, model_slug)
            self.cross_mem = CrossEpisodeMemory(self.cross_mem_dir)
        else:
            self.cross_mem_dir = None
            self.cross_mem = None
        
        # Simple loop detection buffers
        self._recent_states = []
        self._recent_actions = []
        self._recent_scores = []
        self._dacs_candidate_bonus = {}
        self._state_action_failures = {}
        self._state_action_attempts = {}
        self._recorded_outcomes = 0
        self._last_reward = 0
        self._inventory_items = []
        self._inventory_changed_recently = False
        self._last_taken_object = ""
        self._state_success_actions = {}
    
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)  # Remove oldest entry if exceeding max_memory
    
    def _format_memory_for_prompt(self):
        if not self.memory:
            return ""
            
        memory_text = "MEMORY (Recent few states and agent's responses):\n"
        for i, entry in enumerate(self.memory):
            memory_text += f"Memory {i+1}:\n"
            memory_text += f"STATE: {entry['state']}\n"
            if entry['response']:
                memory_text += f"AGENT'S RESPONSE: {entry['response']}\n"
        
        return memory_text

    def calculate_ucb(self, node_idx: int) -> float:
        """
        Calculate the UCB value for a node
        UCB = node's score + c * alpha^depth * sqrt(log(N_total / (1+N_children)))
        Where N_total is the length of the nodes list, alpha is the depth decay factor,
        And N_children is the number of children of this node
        """
        node = self.nodes[node_idx]
        num_children = len(node["children_idxs"])
        total_nodes = max(2, len(self.nodes))

        # If frozen (we detected a win), stop exploration entirely
        if self.is_frozen or (self.freeze_on_win and self.best_score is not None and self.win_freeze_threshold and self.best_score >= self.win_freeze_threshold):
            return node["score"]

        c = self.args.exploration_constant
        return node["score"] + c * (self.args.depth_constant ** node["depth"]) * math.sqrt(math.log(total_nodes) / (1 + num_children))

    def _calculate_routed_ucb(self, node_idx: int) -> float:
        bonus_weight = float(getattr(self.args, "dacs_ucb_bonus_weight", 0.75) or 0.0)
        if not getattr(self.args, "enable_dacs", False) or bonus_weight <= 0:
            return self.calculate_ucb(node_idx)
        return self.calculate_ucb(node_idx) + bonus_weight * self._dacs_candidate_bonus.get(node_idx, 0.0)
    
    def start_episode(self):
        """
        Select a node based on UCB, evolve the guiding prompt, and create a new node.
        If no nodes exist, use the initial guiding prompt.
        New node's score and history will be updated at the end of the episode.
        """
        self.memory = []
        self.game_history = []
        self._recent_states = []
        self._recent_actions = []
        self._recent_scores = []
        self._state_action_failures = {}
        self._state_action_attempts = {}
        self._recorded_outcomes = 0
        self._last_reward = 0
        self._inventory_items = []
        self._inventory_changed_recently = False
        self._last_taken_object = ""
        self._state_success_actions = {}

        should_exploit_best = False
        # Condition 1: explicit win-freeze threshold reached in prior runs
        if self.freeze_on_win and self.win_freeze_threshold and self.best_score >= self.win_freeze_threshold:
            should_exploit_best = True
        # Condition 2: auto frozen due to detected win (no manual threshold needed)
        if self.is_frozen:
            should_exploit_best = True
        # Condition 3: large drop from best in last run
        elif self.force_best_after_drop and self.last_episode_score is not None and self.best_score - self.last_episode_score >= self.drop_threshold:
            should_exploit_best = True
        # Condition 4: stabilize a freshly discovered strong configuration and
        # recover quickly after a meaningful regression, without game-specific thresholds.
        if (
            self.best_node_idx is not None
            and self.last_episode_score is not None
            and self.best_score > 0
        ):
            if self.last_episode_score >= self.best_score:
                should_exploit_best = True
            else:
                meaningful_drop = max(2.0, 0.25 * abs(float(self.best_score)))
                if float(self.best_score) - float(self.last_episode_score) >= meaningful_drop:
                    should_exploit_best = True
        
        # If we just unfroze because the frozen prompt failed to win, we must evolve next
        if self.just_unfroze_due_to_missed_win:
            should_exploit_best = False

        if len(self.nodes) > 0:
            if should_exploit_best and self.best_node_idx is not None:
                parent_idx = self.best_node_idx
                parent_node = self.nodes[parent_idx]
                self.guiding_prompt = parent_node["prompt"]
                self.code = parent_node["code"]
                print(f"[OurAgent] Exploiting best node {parent_idx} (score={self.best_score}). No evolution.")
            else:
                candidate_indices = list(range(len(self.nodes)))
                # DACS: optionally filter / reorder the candidate parent pool
                # before the original UCB selector picks from it. On any error
                # we transparently fall back to the full candidate list.
                if getattr(self.args, 'enable_dacs', False):
                    candidate_indices = self._dacs_filter_candidates(candidate_indices)
                parent_idx = max(candidate_indices, key=lambda i: self._calculate_routed_ucb(i))
                parent_node = self.nodes[parent_idx]
                print(
                    f"Parent node {parent_idx} at depth {parent_node['depth']} "
                    f"with UCB score: {self.calculate_ucb(parent_idx)} "
                    f"and routed score: {self._calculate_routed_ucb(parent_idx)}"
                )
                neg_block = self._format_negative_block_for_evolve() if self.enable_cross_mem else ""
                self.guiding_prompt, self.code = self._evolve(parent_node["prompt"], parent_node["code"], parent_node["game_history"], neg_block=neg_block)
                print(f"Evolved guiding prompt and code: '{self.guiding_prompt}'")
                print(self.code)
        else:
            print(f"Using initial prompt and code: '{self.guiding_prompt}'")
            print(self.code)
            parent_idx = -1
            
        # Create a new node with evolved prompt as child of selected node
        new_node = {
            "prompt": self.guiding_prompt,
            "code": self.code,
            "depth": 0 if parent_idx < 0 else self.nodes[parent_idx]["depth"] + 1,
            "score": -1,
            "parent_idx": parent_idx,
            "children_idxs": [],
            "game_history": []
        }
            
        # Update parent's children list
        if parent_idx >= 0:
            self.nodes[parent_idx]["children_idxs"].append(len(self.nodes))
        self.nodes.append(new_node)
        
        # Reset this one-shot switch after we've chosen to evolve
        self.just_unfroze_due_to_missed_win = False
        
        # self.add_to_memory("=== START OF GAME ===", "")

    def end_episode(self, state, score):
        """
        End an episode: update the current node's score and game history.
        """
        # self.add_to_memory("=== END OF GAME ===", "")
        self._add_to_game_history(state, '', '', '')
        self.nodes[-1]["game_history"] = self._format_game_history(self.game_history)
        self.nodes[-1]["score"] = score
        self.last_episode_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_node_idx = len(self.nodes) - 1
            print(f"[OurAgent] New best score {self.best_score} at node {self.best_node_idx}.")

        # Evaluate victory on final observation and update frozen state
        victory = self._detect_victory_from_observation(state)
        self.last_episode_was_victory = victory
        if self.auto_freeze_on_win and victory:
            self.is_frozen = True
            # If no manual threshold provided, adopt the winning score as threshold for transparency
            if not self.win_freeze_threshold:
                self.win_freeze_threshold = score
                self._auto_set_threshold = True
            print(f"[OurAgent] Victory detected. Freezing exploration. Threshold set to {self.win_freeze_threshold}.")
        else:
            # If we were frozen but failed to win this episode, unfreeze and resume evolution
            if self.is_frozen and not victory:
                self.is_frozen = False
                if self._auto_set_threshold:
                    self.win_freeze_threshold = 0
                    self._auto_set_threshold = False
                # Ensure next episode evolves instead of re-exploiting immediately
                self.just_unfroze_due_to_missed_win = True
                print("[OurAgent] Missed win with frozen prompt. Unfreezing and resuming evolution.")

        # Persist any detected loop as negative memory at episode end (best-effort)
        if self.enable_cross_mem:
            loop_segment = self._detect_loop_segment()
            if loop_segment is not None:
                states_seg, actions_seg = loop_segment
                self.cross_mem.add_negative(states_seg, actions_seg, reason='loop_zero_gain', extra={'episode_final_score': score})

    def get_prompts(self, state_node):
        memory_text = self._format_memory_for_prompt()
        
        # Extract current state using the state extractor
        extracted_state = self._extract_current_state()
        state_summary = ""
        if extracted_state:
            state_summary = f"GAME STATE SUMMARY: {extracted_state}"

        # Few-shot from cross-episode positives
        few_shot = self._format_few_shot_from_cross_mem(state_node.state) if self.enable_cross_mem else "(disabled)"

        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
{state_summary}\n\nYour memory of the recent states and actions is: {memory_text}\n\n
Here are 2-3 successful examples from similar situations (learned across episodes):
{few_shot}

Your current state is: {state_node.state} \n\n
Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt, extracted_state

    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node):
        self._record_previous_action_outcome(state_node.state)
        sys_prompt, user_prompt, extracted_state = self.get_prompts(state_node)
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=400,
            temperature=self.args.llm_temperature,
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            action_text = self._parse_llm_response(full_response)
        else:
            print(f"Warning: LLM API call might have failed or returned empty. Defaulting action.")
            full_response = ""
            action_text = "look" # Default action

        action_text = self._stabilize_action(action_text, state_node.state)
            
        self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response, extracted_state)
        
        # Track for loop detection
        self._recent_states.append(state_node.state)
        self._recent_actions.append(action_text)
        # score will be added on step via update_game_history_reward
        
        return action_text.strip(), full_response

    def _parse_llm_response(self, full_response: str):
        """
        Parses the LLM's full string response to extract action.
        """
        action_text = "look" # Default action

        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        try:
            for line in lines:
                if line.upper().startswith("ACTION:"):
                    action_text = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Response was: '{full_response}'")

        return action_text

    def _stabilize_action(self, action_text: str, current_state: str) -> str:
        action = self._normalize_command(action_text)
        action = self._canonicalize_command(action)
        effective_state = self._effective_state_for_action(current_state)
        state_key = self._state_key(effective_state)
        action_key = self._action_key(action)

        if not action:
            return self._choose_escape_action(effective_state, reason="empty_action")

        attempts = self._state_action_attempts.get((state_key, action_key), 0)
        failures = self._state_action_failures.get((state_key, action_key), 0)
        recent_same_state_actions = [
            a for s, a in zip(self._recent_states[-12:], self._recent_actions[-12:])
            if self._state_key(self._effective_state_for_action(s)) == state_key
        ]
        look_count = sum(1 for a in recent_same_state_actions if self._action_key(a).startswith("look"))
        repeated_same = sum(1 for a in recent_same_state_actions if self._action_key(a) == action_key)
        stagnant = self._is_stagnating()

        if failures > 0 and (stagnant or repeated_same > 0):
            return self._choose_escape_action(effective_state, reason="failed_action", avoid=action_key)
        if attempts > 2 and (stagnant or action_key.startswith(("look", "inventory", "wait"))):
            return self._choose_escape_action(effective_state, reason="repeated_attempt", avoid=action_key)
        if action_key.startswith("look") and (look_count >= 4 or (stagnant and look_count >= 2)):
            return self._choose_escape_action(effective_state, reason="look_loop", avoid=action_key)
        if repeated_same >= 2 and stagnant:
            return self._choose_escape_action(effective_state, reason="same_action_loop", avoid=action_key)

        self._state_action_attempts[(state_key, action_key)] = attempts + 1
        return action

    def _record_previous_action_outcome(self, current_state: str):
        if self._recorded_outcomes >= len(self._recent_actions):
            return
        if not self._recent_actions or not self._recent_states:
            return
        prev_state = self._recent_states[-1]
        prev_action = self._recent_actions[-1]
        self._update_inventory_signals(current_state, prev_action)
        prev_state_key = self._state_key(self._effective_state_for_action(prev_state))
        action_key = self._action_key(prev_action)
        current_key = self._state_key(current_state)
        prev_key = self._state_key(prev_state)
        failed = (
            self._last_reward <= 0
            and (
                self._looks_like_invalid_feedback(current_state)
                or current_key == prev_key
                or action_key.startswith(("look", "inventory", "wait"))
            )
        )
        if failed:
            self._state_action_failures[(prev_state_key, action_key)] = (
                self._state_action_failures.get((prev_state_key, action_key), 0) + 1
            )
        elif current_state and not self._looks_like_invalid_feedback(current_state) and not self._looks_like_transient_feedback(current_state):
            current_effective_key = self._state_key(current_state)
            if current_effective_key != prev_state_key or self._last_reward > 0:
                actions = self._state_success_actions.setdefault(prev_state_key, [])
                if action_key and action_key not in actions:
                    actions.append(action_key)
        self._recorded_outcomes = len(self._recent_actions)

    def _normalize_command(self, action_text: str) -> str:
        if not action_text:
            return ""
        action = str(action_text).strip()
        action = action.splitlines()[0].strip()
        action = re.sub(r"^(action|command)\s*:\s*", "", action, flags=re.IGNORECASE).strip()
        action = re.split(r"\s*(?:;|\bthen\b|\band then\b|,)\s*", action, maxsplit=1, flags=re.IGNORECASE)[0]
        action = re.sub(r"\s+", " ", action).strip(" .\"'`")
        return action.lower()

    def _canonicalize_command(self, action: str) -> str:
        if not action:
            return ""
        parts = action.split()
        if not parts:
            return ""
        if parts[0] == "go" and len(parts) >= 2:
            if parts[1] in self._direction_words():
                return parts[1]
            return " ".join(parts[:2])
        if parts[0] in ("enter", "climb") and len(parts) > 2:
            return " ".join(parts[:2])
        if parts[0] in ("examine", "search", "take", "get", "use", "open", "push") and len(parts) > 2 and parts[1] in ("at", "the", "a", "an"):
            return " ".join([parts[0], parts[2]])
        if parts[0] == "look" and len(parts) > 1 and parts[1] not in ("in", "inside", "under", "behind"):
            return "examine " + " ".join(parts[1:3])
        return " ".join(parts[:5])

    def _choose_escape_action(self, state: str, reason: str = "", avoid: str = "") -> str:
        state_key = self._state_key(state)
        tried = {
            self._action_key(a)
            for s, a in zip(self._recent_states, self._recent_actions)
            if self._state_key(self._effective_state_for_action(s)) == state_key
        }
        if avoid:
            tried.add(avoid)

        candidates = []
        primary_dirs = self._mentioned_directions(state)
        if not primary_dirs:
            primary_dirs = ["north", "south", "east", "west"]
        candidates.extend(primary_dirs)
        candidates.extend([d for d in ["north", "south", "east", "west"] if d not in candidates])
        if any(d in primary_dirs for d in ("up", "down")):
            candidates.extend([d for d in ["up", "down"] if d not in candidates])
        for obj in self._visible_object_candidates(state):
            candidates.append(f"examine {obj}")
            candidates.append(f"search {obj}")
            candidates.append(f"take {obj}")
        if self._inventory_changed_recently or self._is_stagnating():
            for obj in self._current_inventory_candidates():
                candidates.append(f"examine {obj}")
                candidates.append(f"use {obj}")
        candidates.extend(["look", "inventory"])

        for candidate in candidates:
            c = self._canonicalize_command(candidate)
            key = self._action_key(c)
            if not c or key in tried:
                continue
            if self._state_action_failures.get((state_key, key), 0) > 0:
                continue
            self._state_action_attempts[(state_key, key)] = (
                self._state_action_attempts.get((state_key, key), 0) + 1
            )
            if getattr(self.args, "debug_info", False) or getattr(self.args, "dacs_debug", False):
                print(f"[OurAgent] Action escape ({reason}): {c}")
            if self._inventory_changed_recently and any(c.endswith(f" {obj}") for obj in self._current_inventory_candidates()):
                self._inventory_changed_recently = False
            return c

        fallback = self._recovery_escape_action(state, state_key, tried, avoid)
        if getattr(self.args, "debug_info", False) or getattr(self.args, "dacs_debug", False):
            print(f"[OurAgent] Action escape fallback ({reason}): {fallback}")
        return fallback

    def _recovery_escape_action(self, state: str, state_key: str, tried, avoid: str = "") -> str:
        candidates = []
        candidates.extend(self._state_success_actions.get(state_key, []))
        mentioned = self._mentioned_directions(state)
        candidates.extend([d for d in mentioned if d not in candidates])
        candidates.extend([d for d in ["north", "south", "east", "west"] if d not in candidates])
        if any(d in mentioned for d in ("up", "down")):
            candidates.extend([d for d in ["up", "down"] if d not in candidates])
        candidates.extend(["look", "inventory"])
        last_action = self._action_key(self._recent_actions[-1]) if self._recent_actions else ""
        recent_actions = [self._action_key(a) for a in self._recent_actions[-12:]]
        plateau = self._is_zero_score_plateau(12)
        for candidate in candidates:
            c = self._canonicalize_command(candidate)
            key = self._action_key(c)
            if not c or key == avoid or key == last_action:
                continue
            if plateau and key in recent_actions:
                continue
            if key == "inventory" and recent_actions.count("inventory") >= 2:
                continue
            failures = self._state_action_failures.get((state_key, key), 0)
            if key in self._state_success_actions.get(state_key, []):
                return c
            if key in tried and failures > 0:
                continue
            if failures <= 0:
                return c
        final_candidates = ["look", "south", "east", "north", "west", "inventory"]
        for candidate in final_candidates:
            key = self._action_key(candidate)
            if key == avoid or key == last_action:
                continue
            if key == "inventory" and recent_actions.count("inventory") >= 2:
                continue
            return candidate
        return "look"

    def _is_zero_score_plateau(self, steps: int = 12) -> bool:
        if len(self._recent_scores) < steps:
            return False
        tail = self._recent_scores[-steps:]
        return len(set(tail)) <= 1

    def _effective_state_for_action(self, current_state: str) -> str:
        if current_state and not self._looks_like_invalid_feedback(current_state) and not self._looks_like_transient_feedback(current_state):
            return current_state
        for prev in reversed(self._recent_states):
            if prev and not self._looks_like_invalid_feedback(prev) and not self._looks_like_transient_feedback(prev):
                return prev
        return current_state or ""

    def _state_key(self, state: str) -> str:
        text = re.sub(r"\s+", " ", (state or "").lower()).strip()
        return text[:360]

    def _action_key(self, action: str) -> str:
        return re.sub(r"\s+", " ", (action or "").lower()).strip()

    def _looks_like_invalid_feedback(self, state: str) -> bool:
        text = (state or "").lower()
        markers = [
            "you can't",
            "can't see",
            "can't go",
            "cannot",
            "nothing happens",
            "not a verb",
            "don't understand",
            "i beg your pardon",
            "what do you want to",
            "find nothing of interest",
            "no reply",
            "that's not something",
            "you find nothing of interest",
            "not available",
            "doesn't seem interested",
        ]
        return any(m in text for m in markers)

    def _looks_like_transient_feedback(self, state: str) -> bool:
        text = re.sub(r"\s+", " ", (state or "").lower()).strip()
        if not text:
            return False
        if text in {"taken.", "taken", "dropped.", "dropped", "done.", "done", "ok.", "ok"}:
            return True
        return text.startswith("you are carrying")

    def _is_stagnating(self) -> bool:
        if len(self._recent_scores) >= 6 and len(set(self._recent_scores[-6:])) <= 1:
            return True
        if len(self._recent_states) >= 5:
            keys = [self._state_key(self._effective_state_for_action(s)) for s in self._recent_states[-5:]]
            if len(set(keys)) <= 2:
                return True
        return False

    def _direction_words(self):
        return ["north", "south", "east", "west", "up", "down"]

    def _mentioned_directions(self, state: str):
        text = (state or "").lower()
        dirs = []
        direction_re = r"(north|south|east|west|up|down)"
        explicit_patterns = [
            rf"\b(?:exit|exits|door|doors|path|paths|passage|passages|way|ways)\b[^.\n]{{0,60}}\b(?:to|towards|toward|leads?|goes?|runs?)\b[^.\n]{{0,20}}\b{direction_re}\b",
            rf"\b(?:to|towards|toward)\s+the\s+{direction_re}\b[^.\n]{{0,50}}\b(?:is|are|lies?|stands?|door|path|passage|way|exit)\b",
            rf"\b{direction_re}\b[^.\n]{{0,35}}\b(?:exit|door|path|passage|way)\b",
        ]
        for pattern in explicit_patterns:
            for match in re.finditer(pattern, text):
                direction = next((g for g in match.groups() if g in self._direction_words()), None)
                if direction and direction not in dirs:
                    dirs.append(direction)
        aliases = {
            "north": ["north", "northern"],
            "south": ["south", "southern"],
            "east": ["east", "eastern"],
            "west": ["west", "western"],
            "up": ["up", "above", "stair", "stairs", "staircase"],
            "down": ["down", "below"],
        }
        for direction, words in aliases.items():
            if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
                dirs.append(direction)
        return dirs

    def _visible_object_candidates(self, state: str):
        text = (state or "").lower()
        candidates = []
        for match in re.finditer(
            r"(?:you can (?:also )?see|there is|there are|contains?)\s+([^.\n]{1,120})",
            text,
        ):
            candidates.extend(self._object_heads_from_phrase(match.group(1)))
        blocked = self._object_stopwords()
        out = []
        for c in candidates:
            if c not in blocked and c not in out and len(c) > 2:
                out.append(c)
            if len(out) >= 6:
                break
        return out

    def _object_heads_from_phrase(self, text: str):
        cleaned = re.sub(r"\([^)]*\)", " ", text or "")
        cleaned = re.sub(r"\b(?:here|nearby|around|visible|lying|standing|sitting)\b", " ", cleaned)
        pieces = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", cleaned)
        out = []
        for piece in pieces:
            tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", piece.lower())
            meaningful = [t for t in tokens if t not in self._object_stopwords()]
            if not meaningful:
                continue
            head = meaningful[-1]
            if head not in out:
                out.append(head)
        return out

    def _object_stopwords(self):
        return set(self._direction_words()) | {
            "you", "are", "the", "and", "with", "your", "this", "that", "here",
            "there", "room", "game", "score", "nothing", "interest", "way",
            "some", "any", "one", "two", "three", "four", "five", "many",
            "ornately", "carved", "golden", "large", "small", "medium", "new",
            "looking", "pretty", "tough", "simple", "single", "long", "wooden",
            "old", "stone", "metal", "brass", "neat", "half", "finished",
            "perhaps", "mostly", "empty", "front", "back", "side", "kind",
            "direction", "description", "building", "office", "entrance",
        }

    def _current_inventory_candidates(self):
        out = []
        if self._last_taken_object:
            out.append(self._last_taken_object)
        for item in self._inventory_items:
            if item not in out:
                out.append(item)
        return out[:4]

    def _update_inventory_signals(self, state: str, action: str):
        text = re.sub(r"\s+", " ", (state or "").lower()).strip()
        action = self._action_key(action)
        if text.startswith("you are carrying"):
            items_part = text.split("you are carrying", 1)[1]
            if "nothing" in items_part:
                self._inventory_items = []
            else:
                parsed = self._object_heads_from_phrase(items_part.replace(":", " "))
                if parsed:
                    self._inventory_items = parsed[:6]
            return
        if text in {"taken.", "taken"} and action:
            parts = action.split()
            if parts and parts[0] in {"take", "get", "pick"} and len(parts) >= 2:
                obj = parts[-1]
                if obj not in self._object_stopwords():
                    self._last_taken_object = obj
                    if obj not in self._inventory_items:
                        self._inventory_items.insert(0, obj)
                    self._inventory_changed_recently = True
    
    def _add_to_game_history(self, state, action, full_response, extracted_state, reward=None, score=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "extracted_state": extracted_state,
            "reward": reward,
            "score": score
        })
    
    def update_game_history_reward(self, reward, score):
        """Update the last entry in game history with reward and score"""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score
            self._last_reward = reward or 0
            if self.enable_cross_mem:
                # For positives: if delta_score>0, persist (state->action)
                if len(self._recent_scores) == 0:
                    prev = 0
                else:
                    prev = self._recent_scores[-1]
                delta = (score or 0) - (prev or 0)
                self._recent_scores.append(score or 0)
                if delta > 0:
                    last_state = self._recent_states[-1] if self._recent_states else ""
                    last_action = self._recent_actions[-1] if self._recent_actions else ""
                    try:
                        self.cross_mem.add_positive(last_state, last_action, delta_score=delta, extra={"reward": reward, "score": score})
                    except Exception:
                        pass

    
    def _format_game_history(self, history):
        
        history_str = "GAME HISTORY:\n"
        for i, entry in enumerate(history):
            history_str += f"Step {i+1}:\n"
            history_str += f"STATE: {entry['state']}\n"
            if 'extracted_state' in entry and entry['extracted_state']:
                history_str += f"EXTRACTED STATE: {entry['extracted_state']}\n"
            if 'full_response' in entry and entry['full_response']:
                history_str += f"AGENT'S FULL RESPONSE: {entry['full_response']}\n"
            if 'action' in entry and entry['action']:
                history_str += f"ACTION TAKEN: {entry['action']}\n"
            if entry.get('reward') is not None:
                history_str += f"REWARD: {entry['reward']}\n"
            if entry.get('score') is not None:
                history_str += f"SCORE: {entry['score']}\n"
            history_str += "------------\n"
        
        return history_str
    
    def _evolve(self, cur_prompt, cur_code, cur_history_str, neg_block: str = ""):
        print(f"\nEvolving prompt. Current prompt: '{cur_prompt[:80]}...'\n\nCurrent state extractor code: '{cur_code[:80]}...'\n\n")

        sys_prompt_evo = "You are an expert at text adventure games. Your goal is to analyze the existing prompt, state extractor code (i.e. python code which outputs a concise summary of the game state to help the agent, using the game history as input), and game history, and generate a better prompt and state extractor code that will help an LLM agent achieve higher scores. Don't be overly concise; ignore oververbosity penalties."
        
        negative_section = ""
        if neg_block and self.enable_cross_mem:
            negative_section = f"\n\nAVOID THE FOLLOWING FAILURE PATTERNS (derived from prior episodes):\n{neg_block}\n\n"
        
        self.user_prompt_evo = f'''
Generate a new improved guiding prompt and state extractor code for a text adventure game agent. 

The LLM agent used the following guiding prompt (which may not be accurate; rewrite it completely if needed):
"{cur_prompt}"

Here is the history of that game session:
--- GAME HISTORY START ---
{cur_history_str}
--- GAME HISTORY END ---
{negative_section}
PART 1: Generate a new improved guiding prompt. Consider:
1. Identify useful actions that led to increases in score, or needed for progressing the game, ignoring useless actions. Give step-by-step instructions to perform these actions. ONLY give instructions that were strictly necessary for progressing the game or give rewards. Do not suggest possible future actions as they may not be correct.
2. Discourage actions that led to negative outcomes, getting stuck, or unproductive for too long.
3. When reaching the limit based on current game knowledge, list lightly searched rooms and generic exploration categories. Do NOT promote unseen objects, unseen keys/codes, unseen NPCs, or unobserved mechanics into the main plan. A future action may be specific only if its object appeared in the environment text, inventory text, or a prior successful action; otherwise describe it as a low-priority generic guess.

PART 2: Generate a state extractor Python code in a <code>...</code> block that analyzes the game history log and summarizes what milestones the agent has completed so far, i.e. significant progression towards completing the game. This should be a Python function that:
1. Take the game history as input (a string with the log of the game states and agent's actions)
2. Extracts key information about the agent's current milestones reached (relevant to making progress in the game)
3. Returns a summary string of the current state (e.g., "Opened blue door.")

The state extractor can be tailored to the current game; it does not have to work for other games. Avoid complex code to avoid bugs; use simple checks like for particular *strings from the game environment* indicating that a certain milestone was completed, not the agent's actions or commands, since there are usually many ways to express an action. Be careful with using int() as numeric values are sometimes given in words. 
<code>
def extract_state(game_history):
    if "The blue door opens" in game_history:
        return "Opened blue door."
    else:
        return "Unknown state."
</code>

Format your response as follows with NO additional text. The function name MUST be extract_state, and should contain no comments.

[Your generated prompt here]
<code>
def extract_state(game_history):
    # [Return a string summarizing the current state]
</code>
'''
        try:
            response = chat_completion_with_retries(
                model=self.args.evolution_llm_model,
                sys_prompt=sys_prompt_evo,
                prompt=self.user_prompt_evo,
                max_tokens=3000,
                temperature=self.args.evol_temperature,
            )
            content = ""
            if response and getattr(response, "choices", None) and response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            full_response = content.strip()
            
            ret_prompt, ret_code = cur_prompt, cur_code
            new_prompt, new_code = self._parse_evolution_response(full_response)
            
            if new_prompt and len(new_prompt) > 10:
                if self._is_speculative_evolved_prompt(new_prompt, cur_history_str):
                    print("Evolution LLM returned over-speculative prompt, keeping current prompt")
                else:
                    ret_prompt = self._repair_evolved_prompt(new_prompt)
            else:
                print("Evolution LLM returned empty/short response, keeping current prompt")
            if new_code and self._validate_state_extractor(new_code) and len(new_code) > 10:
                ret_code = new_code
            else:                
                print("Evolution LLM returned invalid state extractor code, keeping current code")
            return ret_prompt, ret_code
                
        except Exception as e:
            print(f"Error during prompt evolution LLM call: {e}")
            return cur_prompt, cur_code  # Return current prompt if evolution fails
    
    def _parse_evolution_response(self, response):
        """
        Parse the response from the evolutionary LLM to extract:
        1. The new prompt
        2. The state extractor code
        
        Returns:
            tuple: (new_prompt, state_extractor_code)
        """
        # Extract state extractor code
        state_extractor_code = ""
        if "<code>" in response and "</code>" in response:
            code_start = response.find("<code>") + len("<code>")
            code_end = response.find("</code>")
            if code_start < code_end:
                state_extractor_code = response[code_start:code_end].strip()
                # Remove the code part from the response to get the prompt
                response = response[:response.find("<code>")].strip()
        
        # The remaining text is the prompt
        new_prompt = response.strip()
        
        return new_prompt, state_extractor_code

    def _repair_evolved_prompt(self, prompt: str) -> str:
        guard = (
            "Evidence-first control: prioritize actions that previously increased score, "
            "reached a new observation/location, interacted with a visible object, or used an "
            "inventory item that was actually observed. Treat commands involving unseen objects, "
            "unseen locks, unseen keys/codes, unseen NPCs, or inferred mechanics as low-priority "
            "guesses only after systematic navigation, visible-object checks, and inventory checks. "
            "Do not repeat a zero-reward action after invalid feedback unless new state text appears."
        )
        text = re.sub(r"\s+", " ", (prompt or "")).strip()
        if not text:
            return guard
        if "Evidence-first control:" in text:
            return text
        return f"{guard}\n\n{text}"

    def _is_speculative_evolved_prompt(self, prompt: str, history: str) -> bool:
        prompt_text = (prompt or "").lower()
        history_text = (history or "").lower()
        if not prompt_text:
            return True
        speculative_terms = [
            "key", "keys", "code", "codes", "password", "unlock", "locked",
            "hidden", "secret", "npc", "character", "person", "switch",
            "lever", "drawer", "note",
        ]
        unseen_terms = [t for t in speculative_terms if t in prompt_text and t not in history_text]
        if len(unseen_terms) >= 2:
            return True
        if ("exact steps" in prompt_text or "follow these exact" in prompt_text) and unseen_terms:
            return True
        return False
    
    def _validate_state_extractor(self, state_extractor_code):
        
        if not state_extractor_code:
            print("No valid state extractor code provided.")
            return False

        try:
            namespace = {}
            expected_fn_header = "def extract_state(game_history):"
            if expected_fn_header not in state_extractor_code:
                state_extractor_code = expected_fn_header + "\n    " + \
                    "# Default implementation if provided code lacks the proper function\n    " + \
                    "return \"No specific state extracted\"\n\n" + state_extractor_code
            
            exec(state_extractor_code, namespace)
            if "extract_state" in namespace and callable(namespace["extract_state"]):
                try:
                    namespace["extract_state"]("")
                    print("Updated state extractor code successfully.")
                    return True
                except Exception as e:
                    print(f"State extractor code failed basic test: {e}")
                    return False
            else:
                print("State extractor code does not contain valid extract_state function.")
                return False
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def _extract_current_state(self):
        """
        Extract the current state from the game history using the state extractor code.
        Returns:
            str: The extracted state description or empty string if extraction fails.
        """
        if not hasattr(self, 'code') or not self.code:
            return ""
        
        try:
            namespace = {}
            exec(self.code, namespace)
            history_str = self._format_game_history(self.game_history)
            extracted_state = namespace["extract_state"](history_str)
            return str(extracted_state) if extracted_state else ""
        except Exception as e:
            print(f"Error extracting state: {e}")
            return ""

    def _detect_victory_from_observation(self, final_observation_text: str) -> bool:
        if not final_observation_text or not isinstance(final_observation_text, str):
            return False
        text = final_observation_text.lower()
        # Heuristics for victory/credits screens common in text adventures
        victory_keywords = [
            "you have won", "victory", "congratulations", "congrats", "credits", "the end", "you win",
            # game-specific hints seen in logs
            "info room", "promoted", "win 310", "win 360"
        ]
        defeat_keywords = ["die", "died", "death", "killed", "game over", "defeat"]
        if any(k in text for k in victory_keywords) and not any(k in text for k in defeat_keywords):
            return True
        return False

    # -------------------- Cross-episode helpers --------------------
    def _format_few_shot_from_cross_mem(self, current_state: str) -> str:
        try:
            examples = self.cross_mem.retrieve_similar(current_state, k=3)
        except Exception:
            examples = []
        if not examples:
            return "(none)"
        lines = []
        for ex in examples:
            lines.append(f"STATE: {ex.get('state','')[:400]}")
            lines.append(f"ACTION: {ex.get('action','')}")
            lines.append(f"GAIN: +{ex.get('delta_score',0)}")
            lines.append("---")
        return "\n".join(lines)

    def _detect_loop_segment(self):
        # Heuristic: look for last 20 steps repeating between two observations or zero-gain plateau for >= 10 steps
        if len(self._recent_states) < 12 or len(self._recent_scores) < 12:
            return None
        # Zero-gain plateau
        plateau = all((self._recent_scores[-i-1] == self._recent_scores[-1]) for i in range(10))
        if plateau:
            seg_len = min(40, len(self._recent_states))
            return self._recent_states[-seg_len:], self._recent_actions[-seg_len:]
        # Simple immediate alternation a<->b pattern
        a = self._recent_states[-1]
        b = self._recent_states[-2]
        alt = True
        for i in range(3, min(20, len(self._recent_states))):
            if (i % 2 == 0 and self._recent_states[-i] != b) or (i % 2 == 1 and self._recent_states[-i] != a):
                alt = False
                break
        if alt:
            seg_len = min(40, len(self._recent_states))
            return self._recent_states[-seg_len:], self._recent_actions[-seg_len:]
        return None

    def _format_negative_block_for_evolve(self) -> str:
        try:
            negatives = self.cross_mem.load_negative()
        except Exception:
            negatives = []
        if not negatives:
            return ""
        # Keep last few negative segments
        negatives = negatives[-3:]
        parts = []
        for neg in negatives:
            parts.append(f"Reason: {neg.get('reason','unknown')}, Length: {neg.get('length',0)}")
            # show last few actions only
            actions = neg.get('actions', [])[-10:]
            parts.append("Actions to avoid (tail): " + ", ".join(actions))
        return "\n".join(parts)

    # -------------------- DACS hook --------------------
    def _dacs_filter_candidates(self, candidate_indices):
        """Re-rank Evolver-produced configuration candidates via DACS.

        Each existing tree node is a candidate parent configuration
        (prompt + state-extractor code + accumulated history + score).
        DACS scores them against the most recent trajectory and the
        cross-episode success/failure memory, then returns a diversified
        subset of indices for the original UCB selector to pick from.

        Falls back to the input list on any internal error so the
        experiment never crashes due to DACS.
        """
        try:
            if not candidate_indices:
                return candidate_indices

            latest_node = self.nodes[-1] if self.nodes else None
            transcript = ""
            score_delta = None
            final_score = None
            if latest_node is not None:
                transcript = latest_node.get("game_history", "") or ""
                final_score = latest_node.get("score", None)
                if (
                    self.last_episode_score is not None
                    and self.best_score is not None
                ):
                    try:
                        score_delta = float(self.last_episode_score) - float(
                            self.best_score
                        )
                    except Exception:
                        score_delta = None

            success_memory = []
            failure_memory = []
            if self.enable_cross_mem and self.cross_mem is not None:
                try:
                    success_memory = self.cross_mem.load_positive()[-10:]
                except Exception:
                    success_memory = []
                try:
                    failure_memory = self.cross_mem.load_negative()[-5:]
                except Exception:
                    failure_memory = []

            candidate_configs = [self.nodes[i] for i in candidate_indices]

            top_n = int(getattr(self.args, "dacs_top_n", 8) or 8)
            select_k = int(getattr(self.args, "dacs_select_k", 4) or 4)
            alpha = float(getattr(self.args, "dacs_alpha_relevance", 1.0) or 1.0)
            beta = float(getattr(self.args, "dacs_beta_diversity", 0.5) or 0.5)
            gamma = float(getattr(self.args, "dacs_gamma_risk", 0.7) or 0.7)
            novelty_weight = float(getattr(self.args, "dacs_novelty_weight", 0.35) or 0.35)
            debug = bool(getattr(self.args, "dacs_debug", False))

            result = dacs_mod.select_candidates(
                parent_config=latest_node,
                candidate_configs=candidate_configs,
                transcript=transcript,
                score_delta=score_delta,
                final_score=final_score,
                success_memory=success_memory,
                failure_memory=failure_memory,
                top_n=top_n,
                select_k=select_k,
                alpha_relevance=alpha,
                beta_diversity=beta,
                gamma_risk=gamma,
                novelty_weight=novelty_weight,
                debug=debug,
                game_name=getattr(self.args, "game_name", None),
            )

            if result.get("fallback", False):
                return candidate_indices

            sel = result.get("selected_indices", []) or []
            filtered = [candidate_indices[i] for i in sel if 0 <= i < len(candidate_indices)]
            self._dacs_candidate_bonus = {}
            for row in result.get("scores", []) or []:
                try:
                    native_i = int(row.get("id"))
                    node_i = candidate_indices[native_i]
                    self._dacs_candidate_bonus[node_i] = float(row.get("final_score", 0.0))
                except Exception:
                    continue
            if debug:
                print("[DACS] final routed UCB table:")
                print("  node | ucb | dacs_bonus | routed_ucb | selected_pool")
                selected_pool = set(filtered)
                for node_i in candidate_indices:
                    print(
                        f"  {node_i} | {round(self.calculate_ucb(node_i), 3)} | "
                        f"{round(self._dacs_candidate_bonus.get(node_i, 0.0), 3)} | "
                        f"{round(self._calculate_routed_ucb(node_i), 3)} | "
                        f"{node_i in selected_pool}"
                    )
            return filtered if filtered else candidate_indices
        except Exception as e:
            if getattr(self.args, "dacs_debug", False):
                print(f"[DACS] fallback=True reason={e}")
            return candidate_indices
