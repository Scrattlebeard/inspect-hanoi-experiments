from copy import deepcopy
from dataclasses import dataclass
from inspect_ai.util import StoreModel
import json
import logging
from pydantic import Field, model_validator
import re
import sys
from typing import List, Optional, Tuple, Union
import os

logger = logging.getLogger(__name__)

MAX_MOVES = 25

@dataclass
class InvalidMoveError:
    """Represents an invalid move attempt with detailed error information.

    Attributes:
        move_index: The index of the invalid move in the submitted move list
        move: The actual move that was invalid
        reason: Human-readable explanation of why the move was invalid
    """

    move_index: int
    move: str
    reason: str

    def __str__(self) -> str:
        return f'Invalid move "{self.move}" at index {self.move_index}: {self.reason}'


class TowerOfHanoi(StoreModel):
    n_disks: int = Field(default=3)
    stacks: List[List[int]] = Field(default_factory=lambda: [[3, 2, 1], [], []])
    state_history: List[str] = Field(default_factory=list)

    @staticmethod
    def get_optimal_move_count_for_difficulty(difficulty: int) -> int:
        return (2**difficulty) - 1

    def reset(self, n_disks: int | None = None) -> None:

        if n_disks < 0:
            raise ValueError(f"Number of disks must be non-negative, got {n_disks}")
        
        if n_disks is not None:
            self.n_disks = n_disks
        
        self.stacks = [
            (
                list(range(n_disks, 0, -1)) if n_disks > 0 else []
            ),  # Peg 0: all disks (largest to smallest)
            [],  # Peg 1: empty
            [],  # Peg 2: empty
        ]
        self.state_history = []

    def get_state(self) -> str:
        """Return formatted state string.

        Returns:
            String representation in format:
            "stack 0: 3 (bottom), 2, 1 (top)\nstack 1: (empty)\nstack 2: (empty)"
        """
        lines = []

        for stack_idx, stack_disks in enumerate(self.stacks):
            if not stack_disks:
                lines.append(f"stack {stack_idx}: (empty)")
            elif len(stack_disks) == 1:
                lines.append(f"stack {stack_idx}: {stack_disks[0]}")
            else:
                disk_parts = [f"{stack_disks[0]} (bottom)"]
                if len(stack_disks) > 2:
                    disk_parts.extend(map(str, stack_disks[1:-1]))
                disk_parts.append(f"{stack_disks[-1]} (top)")
                lines.append(f"stack {stack_idx}: {', '.join(disk_parts)}")

        return "\n".join(lines)

    def apply_moves(self, moves: List[List[int]]) -> Union[str, InvalidMoveError]:
        """Apply a list of moves to current state with atomic rollback.

        Move sequences are applied atomically - if any move fails, the entire
        sequence is rolled back and the state remains unchanged.
        """
        if not moves:
            return self.get_state()

        logger.debug(f"Taking state snapshot before applying {len(moves)} moves")
        state_snapshot = deepcopy(self.stacks)
        history_snapshot = self.state_history.copy()

        for move_index, move in enumerate(moves):
            disk, from_stack, to_stack = move
            is_valid, error_message = self.can_move(disk, from_stack, to_stack)
            if is_valid:
                self._execute_move(int(disk), int(from_stack), int(to_stack))
                self.state_history.append(self.get_state())
                logger.debug(f"Applied move {move}: {self.get_state()}")
            else:
                # Rollback to snapshot on execution failure
                logger.debug(f"Move execution failed at index {move_index}, rolling back")
                self.stacks= state_snapshot
                self.state_history = history_snapshot
                return InvalidMoveError(
                    move_index=move_index,
                    move=str(move),
                    reason=f"Failed to execute move: {error_message}",
                )

        logger.debug(f"Successfully applied {len(moves)} moves")
        return self.get_state()

    def _execute_move(self, disk: int, from_stack: int, to_stack: int) -> None:
        """Execute a move if it's legal.

        Args:
            disk: Disk ID to move (1 to n_disks)
            from_stack: Source stack index (0, 1, or 2)
            to_stack: Destination stack index (0, 1, or 2)

        """
        removed_disk = self.stacks[from_stack].pop()
        self.stacks[to_stack].append(removed_disk)
        logger.debug(f"Executed move: disk {disk} from stack {from_stack} to stack {to_stack}")

    def can_move(self, disk: int, from_stack: int, to_stack: int) -> Tuple[bool, str]:
        """Check if a move is legal without executing it.

        Args:
            disk: Disk ID to move (1 to n_disks)
            from_stack: Source stack index (0, 1, or 2)
            to_stack: Destination stack index (0, 1, or 2)

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if move is legal, False otherwise
            - error_message: Empty string if valid, detailed error if invalid
        """

        error = self._validate_move_parameters(disk, from_stack, to_stack)\
             or self._validate_disk_position(disk, from_stack)\
             or self._validate_destination_constraints(disk, to_stack)

        return (False, error) if error else (True, "")

    def get_top_disk(self, stack: int) -> Optional[int]:
        if not self.stacks[stack]:
            return None

        return self.stacks[stack][-1]

    def _validate_stack_index(self, stack: int) -> Optional[str]:
        if not isinstance(stack, int) or stack < 0 or stack >= 3:
            return f"Invalid stack index: {stack}. Must be 0, 1, or 2."

    def _validate_disk_exists(self, disk: int) -> Optional[str]:
        if not isinstance(disk, int) or disk < 1 or disk > self.n_disks:
            return f"Invalid disk ID: {disk}. Must be between 1 and {self.n_disks}."

    def _validate_stacks(self, from_stack: int, to_stack: int) -> Optional[str]:
        if from_stack == to_stack:
            return f"Cannot move from stack {from_stack} to same stack"

    def _validate_move_parameters(self, disk: int, from_stack: int, to_stack: int) -> Optional[str]:
        return self._validate_stack_index(from_stack)\
             or self._validate_stack_index(to_stack)\
             or self._validate_disk_exists(disk)\
             or self._validate_stacks(from_stack, to_stack)

    def _validate_disk_position(self, disk: int, from_stack: int) -> str:
        """Validate that disk is accessible on the source stack."""
        if disk not in self.stacks[from_stack]:
            return (
                f"Disk {disk} is not on stack {from_stack}. Current location: "
                f"{self._find_disk_stack(disk)}"
            )

        top_disk = self.get_top_disk(from_stack)
        if top_disk is None:
                return f"stack {from_stack} is empty, cannot move disk {disk}"

        return f"Cannot move disk {disk}: disk {top_disk} is on top of stack {from_stack}" if top_disk != disk else ""

    def _find_disk_stack(self, disk: int) -> Optional[int]:
        for stack_idx, stack in enumerate(self.stacks):
            if disk in stack:
                return stack_idx
        return None

    def _validate_destination_constraints(self, disk: int, to_stack: int) -> Optional[str]:
        """Validate destination stack constraints."""
        destination_top = self.get_top_disk(to_stack)
        if destination_top is not None and disk > destination_top:
            return (
                f"Cannot place disk {disk} on disk {destination_top}: "
                f"larger disk cannot be placed on smaller disk"
            )

    def is_solved(self) -> bool:
        return (
            len(self.stacks[0]) == 0
            and len(self.stacks[1]) == 0
            and len(self.stacks[2]) == self.n_disks
            and self.stacks[2] == list(range(self.n_disks, 0, -1))
        )

    def parse_moves(self, llm_output: str) -> List[List[int]]:
        """Parse LLM output into move list.

        Supports formats JSON: [[1,0,2], [2,0,1]]
        """
        if not llm_output or not isinstance(llm_output, str):
            raise ValueError("Failed to parse moves from LLM output: Was null or not a string")

        if llm_output == "[]":
            return []

        json_pattern = r"(\[\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])*\s*\])"
        json_matches = re.findall(json_pattern, llm_output)

        if json_matches:
            try:
                json_result = json.loads(json_matches[-1])
                if json_result and isinstance(json_result, list):
                    return json_result
            except Exception:
                pass

        raise ValueError(f"Failed to parse moves from LLM output: {llm_output}")

    def get_move_format(self) -> str:
        """Return string describing expected move format."""
        return "[[disk_id, from_stack, to_stack], ...]"

    def get_state_history(self) -> List[str]:
        """Get history of states for loop detection."""
        return self.state_history.copy()

    def copy(self) -> "TowerOfHanoi":
        new_puzzle = TowerOfHanoi(self.n_disks)
        new_puzzle.stacks = deepcopy(self.stacks)
        new_puzzle.state_history = self.state_history.copy()
        return new_puzzle

    def __str__(self) -> str:
        return f"TowerOfHanoi({self.n_disks} disks):\n{self.get_state()}"

    def get_optimal_move_count(self) -> int:
        return self.n_disks**2 - 1

    def size(self) -> int:
        return self.n_disks

    def _print_status_and_check_extra(self):
        print(self.get_state())
        
        if self.is_solved():
            print("\n🎉 Congratulations! Puzzle solved!")

def main():
    print(f"""RULES:
- There are three stacks (0, 1, 2) and N disks of different sizes
- Only one disk can be moved at a time
- Only the top disk from any stack can be moved
- A larger disk may never be placed on top of a smaller disk
- The goal is to move all disks from stack 0 to stack 2

INTERACTION FORMAT:
- Submit a list of moves in the format: [[disk_id, from_stack, to_stack], ...]
- You can submit up to {MAX_MOVES} moves at a time
- If one of your submitted moves is invalid, you will receive an error message and the puzzle state will roll back - none of the submitted moves will have been applied.""")

    return

if __name__ == "__main__":
    main()