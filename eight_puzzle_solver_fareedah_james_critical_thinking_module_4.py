from simpleai.search import astar, SearchProblem


# Define the goal state of the 8-puzzle
GOAL = '''1-2-3
4-5-6
7-8-e'''

# Function to convert the state from list format to string format


def list_to_string(list_):
    return '\n'.join(['-'.join(row) for row in list_])

# Function to convert the state from string format to list format


def string_to_list(string_):
    return [row.split('-') for row in string_.split('\n')]

# Function to find the location (row and column) of a piece in the puzzle


def find_location(rows, element_to_find):
    for ir, row in enumerate(rows):
        for ic, element in enumerate(row):
            if element == element_to_find:
                return ir, ic


# Precompute the goal positions of each piece to improve performance
goal_positions = {}
rows_goal = string_to_list(GOAL)
for number in '12345678e':
    goal_positions[number] = find_location(rows_goal, number)

# Define the problem class


class EigthPuzzleProblem(SearchProblem):
    def actions(self, state):
        rows = string_to_list(state)
        row_e, col_e = find_location(rows, 'e')

        actions = []
        if row_e > 0:
            actions.append(rows[row_e - 1][col_e])
        if row_e < 2:
            actions.append(rows[row_e + 1][col_e])
        if col_e > 0:
            actions.append(rows[row_e][col_e - 1])
        if col_e < 2:
            actions.append(rows[row_e][col_e + 1])

        return actions

    def result(self, state, action):
        rows = string_to_list(state)
        row_e, col_e = find_location(rows, 'e')
        row_n, col_n = find_location(rows, action)

        rows[row_e][col_e], rows[row_n][col_n] = rows[row_n][col_n], rows[row_e][col_e]

        return list_to_string(rows)

    def is_goal(self, state):
        return state == GOAL

    def cost(self, state1, action, state2):
        return 1

    def heuristic(self, state):
        rows = string_to_list(state)

        distance = 0
        for number in '12345678e':
            row_n, col_n = find_location(rows, number)
            row_n_goal, col_n_goal = goal_positions[number]

            distance += abs(row_n - row_n_goal) + abs(col_n - col_n_goal)

        return distance


# Main function to execute the script
if __name__ == "__main__":
    # Read the initial state from a file named 'puzzle_input.txt'
    with open('puzzle_input.txt', 'r') as file:
        initial_state = file.read().strip()

    # Validate the input format
    input_elements = set(initial_state.replace('-', '').replace('\n', ''))
    expected_elements = set('12345678e')
    if input_elements == expected_elements:
        problem = EigthPuzzleProblem(initial_state)

        result = astar(problem)

        print("\nFound solution:\n")
        for action, state in result.path():
            if action is None:
                print("Initial state:")
            else:
                print(f"Move {action}:")
            print(state)
            print("\n")
    else:
        print("Invalid input. Please ensure all numbers 1-8 and 'e' are present exactly once and in the correct format.")
