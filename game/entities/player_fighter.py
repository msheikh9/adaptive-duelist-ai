"""Human-controlled fighter: translates InputAction → CombatCommitment.

This is the translation boundary between Vocabulary A (InputAction) and
Vocabulary B (CombatCommitment). The AI never sees InputAction.
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment, FSMState
from game.entities.fighter import attempt_commitment, attempt_stop_move
from game.input.input_actions import InputAction
from game.state import FighterState
from config.config_loader import GameConfig


class PlayerController:
    """Processes InputActions for the human player's fighter.

    Tracks held movement keys to handle press/release correctly.
    """

    def __init__(self) -> None:
        self._holding_left = False
        self._holding_right = False

    def process_inputs(self, fighter: FighterState, actions: list[InputAction],
                       config: GameConfig) -> CombatCommitment | None:
        """Process a batch of InputActions for this tick.

        Returns the commitment that was entered (if any), for logging.
        Movement is handled via held-key tracking; attacks/dodge are
        immediate.
        """
        committed: CombatCommitment | None = None

        for action in actions:
            match action:
                case InputAction.PRESS_LEFT:
                    self._holding_left = True
                case InputAction.RELEASE_LEFT:
                    self._holding_left = False
                case InputAction.PRESS_RIGHT:
                    self._holding_right = True
                case InputAction.RELEASE_RIGHT:
                    self._holding_right = False

                case InputAction.PRESS_LIGHT_ATTACK:
                    if attempt_commitment(fighter, CombatCommitment.LIGHT_ATTACK, config):
                        committed = CombatCommitment.LIGHT_ATTACK
                case InputAction.PRESS_HEAVY_ATTACK:
                    if attempt_commitment(fighter, CombatCommitment.HEAVY_ATTACK, config):
                        committed = CombatCommitment.HEAVY_ATTACK
                case InputAction.PRESS_DODGE:
                    if attempt_commitment(fighter, CombatCommitment.DODGE_BACKWARD, config):
                        committed = CombatCommitment.DODGE_BACKWARD

                case InputAction.PRESS_JUMP:
                    # Jump is only valid from ground-level free states
                    if (fighter.fsm_state in (FSMState.IDLE, FSMState.MOVING)
                            and attempt_commitment(fighter, CombatCommitment.JUMP, config)):
                        committed = CombatCommitment.JUMP

                case InputAction.PRESS_BLOCK:
                    if attempt_commitment(fighter, CombatCommitment.BLOCK_START, config):
                        committed = CombatCommitment.BLOCK_START

                case InputAction.RELEASE_BLOCK:
                    if fighter.fsm_state == FSMState.BLOCKING:
                        fighter.fsm_state = FSMState.IDLE
                        fighter.active_commitment = None

                case InputAction.PRESS_SHOOT:
                    # Phase 20: begin startup → charging sequence
                    if attempt_commitment(fighter, CombatCommitment.SHOOT_START, config):
                        committed = CombatCommitment.SHOOT_START

                case InputAction.RELEASE_SHOOT:
                    # Phase 20: release while charging → fire the shot
                    if fighter.fsm_state == FSMState.CHARGING:
                        fighter.fsm_state = FSMState.SHOOT_ACTIVE
                        fighter.fsm_frames_remaining = config.actions.shoot.active_frames
                        fighter.pending_shot = True

        # Handle movement from held keys (only if no attack/dodge was committed)
        if committed is None and fighter.is_free:
            if self._holding_left and not self._holding_right:
                if fighter.active_commitment != CombatCommitment.MOVE_LEFT:
                    attempt_commitment(fighter, CombatCommitment.MOVE_LEFT, config)
            elif self._holding_right and not self._holding_left:
                if fighter.active_commitment != CombatCommitment.MOVE_RIGHT:
                    attempt_commitment(fighter, CombatCommitment.MOVE_RIGHT, config)
            else:
                # No movement keys held, or both held (cancel)
                attempt_stop_move(fighter)

        return committed

    def reset(self) -> None:
        self._holding_left = False
        self._holding_right = False
