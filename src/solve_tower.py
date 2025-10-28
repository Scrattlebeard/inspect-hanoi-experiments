from inspect_ai import Epochs, Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import accuracy,CORRECT, INCORRECT, mean, Scorer, scorer, Score, Target
from inspect_ai.solver import Generate,Solver, solver,TaskState
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store_as
from tower_of_hanoi import TowerOfHanoi

user_message_template = """
This puzzle has {n_disks} disks and you may submit up to {batch_size} moves at a time.

Initial puzzle state:
{puzzle_state}
"""

@tool
def submit_moves() -> Tool:
    
    async def execute(moves: list[list[int]]) -> str:
        """
        Submit a list of moves to the tower of hanoi instance.
        Args:
            moves: A list of moves to submit in the format [[disk_id, from_stack, to_stack], ...].
        Returns:
            A string representing the new state of the puzzle.
        """
        puzzle_state = store_as(TowerOfHanoi)
        return puzzle_state.apply_moves(moves)
    
    return execute

@scorer(metrics=[accuracy()])
def solved() -> Scorer:
    
    async def score(state: TaskState, target: Target) -> Score:
        puzzle_state = store_as(TowerOfHanoi)
        return Score(value=CORRECT) if puzzle_state.is_solved() else Score(value=INCORRECT)
    
    return score

@scorer(metrics=[mean()])
def moves_used() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        puzzle_state = store_as(TowerOfHanoi)
        return Score(value=len(puzzle_state.get_state_history()))
    
    return score

@solver
def setup_tower_instance() -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        puzzle = store_as(TowerOfHanoi)
        puzzle.reset(state.metadata["n_disks"])

        state.messages.append(
            ChatMessageUser(content=user_message_template.format(
                n_disks=state.metadata["n_disks"],
                batch_size=state.metadata["batch_size"],
                puzzle_state=puzzle.get_state()))
        )        
        
        return state
    
    return solve

@task
def tower_of_hanoi() -> Task:
    return Task(
        name="tower_of_hanoi",
        version="0.0.5",
        dataset=json_dataset("tower_of_hanoi_prompt.json"),
        setup=setup_tower_instance(),
        solver=react(
            tools=[submit_moves()]
        ),
        scorer=[solved(), moves_used()],
        epochs=Epochs(1, "mean")
    )