# Student agent: Add your own agent here
from copy import deepcopy
import time
from agents.agent import Agent
from store import register_agent
import math
import numpy as np
import sys


###################################################################################### helper
def check_endgame(chess_board, my_pos, adv_pos):
    board_size = chess_board.shape[0]
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                    moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(my_pos)
    p1_r = find(adv_pos)
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, None
    else:
        if p0_score > p1_score:
            return True, 1
        elif p0_score == p1_score:
            return True, 0
        return True, -1


def expandNode(node):
    possibleStates = node.getAllPossibleStates()
    for newNode in possibleStates:
        newNode.setParent(node)
        node.getLeaf().append(newNode)


def set_barrier(r, c, dir, chess_board):
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Set the barrier to True
    chess_board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = True
    return chess_board


###################################################################################### State class

class State:
    def __init__(self, max_step):
        self.chess_board = None
        self.my_pos = None
        self.adv_pos = None
        self.visitCount = 0
        self.winScore = 0
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # 上，右，下，左
        self.max_step = max_step
        self.playerNo = None
        self.parent = None
        self.leaf = []

    def getLeaf(self):
        return self.leaf

    def getParent(self):
        return self.parent

    def getWinScore(self):
        return self.winScore

    def getBoard(self):
        return self.chess_board

    def getMypos(self):
        return self.my_pos

    def getAdvpos(self):
        return self.adv_pos

    def setParent(self, parent):
        self.parent = parent

    def setBoard(self, chess_board):
        self.chess_board = chess_board

    def setMypos(self, myPos):
        self.my_pos = myPos

    def setAdvpos(self, advPos):
        self.adv_pos = advPos

    def setPlayerNo(self, opponent):
        self.playerNo = opponent

    def getPlayerNo(self):
        return self.playerNo

    def togglePlayer(self):
        self.playerNo = 3 - self.playerNo

    def getOpponent(self):
        return 3 - self.playerNo

    def getVisitCount(self):
        return self.visitCount

    def incrementVisit(self):
        self.visitCount += 1

    def addScore(self, score):
        self.winScore += score

    def checkStatus(self):
        return check_endgame(self.chess_board, self.my_pos, self.adv_pos)

    def getRandomChildNode(self):
        randnum = np.random.randint(0, len(self.leaf))
        return self.leaf[randnum]

    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True
        # Get position of the adversary
        if self.playerNo == 1:
            adv_pos = self.adv_pos
        else:
            adv_pos = self.my_pos
        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached

    def getAllPossibleStates(self):
        allStates = []
        if self.playerNo == 1:
            for i in range(self.chess_board.shape[0]):
                for j in range(self.chess_board.shape[0]):
                    for barrier_dir in range(4):
                        if self.check_valid_step(self.my_pos, (i, j), barrier_dir):
                            new_board = deepcopy(self.chess_board)
                            new_board = set_barrier(i, j, barrier_dir, new_board)
                            new_s = State(self.max_step)
                            new_s.setBoard(new_board)
                            new_s.setMypos((i, j))
                            new_s.setAdvpos(self.adv_pos)
                            new_s.setPlayerNo(2)
                            allStates.append(new_s)
        else:
            for i in range(self.chess_board.shape[0]):
                for j in range(self.chess_board.shape[0]):
                    for barrier_dir in range(4):
                        if self.check_valid_step(self.adv_pos, (i, j), barrier_dir):
                            new_board = deepcopy(self.chess_board)
                            new_board = set_barrier(i, j, barrier_dir, new_board)
                            new_s = State(self.max_step)
                            new_s.setBoard(new_board)
                            new_s.setMypos(self.my_pos)
                            new_s.setAdvpos((i, j))
                            new_s.setPlayerNo(1)
                            allStates.append(new_s)
        return allStates


###################################################################################### student agent class

def evaluate(node):
    chess_board = node.getBoard()
    my_pos = node.getMypos()
    adv_pos = node.getAdvpos()
    check = check_endgame(chess_board, my_pos, adv_pos)
    if check[0]:
        if check[1] == 1:
            return float('inf')
        elif check[1] == -1:
            return float('-inf')
        else:
            return 0

    countMyB = 0
    for dir in range(4):
        i, j = my_pos
        if chess_board[i, j, dir] == False:
            countMyB += 1

    countAdvB = 0
    for dir in range(4):
        i, j = adv_pos
        if chess_board[i, j, dir] == True:
            countAdvB += 1

    return countMyB + countAdvB * 2


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.autoplay = True
        self.stepCount = 0
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.stepCount += 1

        root = State(max_step)
        root.setBoard(chess_board)
        root.setMypos(my_pos)
        root.setAdvpos(adv_pos)
        root.setPlayerNo(1)

        expandNode(root)

        bestScore = float("-inf")
        for childNode in root.getLeaf():
            if evaluate(childNode) >= bestScore:
                bestScore = evaluate(childNode)
                winnerNode = childNode

        r, c = winnerNode.getMypos()
        for direction in range(4):
            if winnerNode.getBoard()[r, c, direction] != root.getBoard()[r, c, direction]:
                break
        return winnerNode.getMypos(), direction

