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
        return False, -1
    else:
        return True, p0_score-p1_score


def findBestNodeWithScore(node):
    bestScore = float('-inf')
    bestNode = None
    for leaf_node in node.getLeaf():
        if leaf_node.getScore() > bestScore:
            bestScore = leaf_node.getScore()
            bestNode = leaf_node
    return bestNode

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
        self.Score = 0
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) #上，右，下，左
        self.max_step = max_step
        self.playerNo = None
        self.parent = None
        self.leaf = []

    def getLeaf(self):
        return self.leaf

    def getParent(self):
        return self.parent

    def getScore(self):
        return self.Score

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
        self.playerNo = 3-self.playerNo

    def getOpponent(self):
        return 3-self.playerNo

    def getVisitCount(self):
        return self.visitCount

    def incrementVisit(self):
        self.visitCount += 1

    def setScore(self, score):
        self.Score = score

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

    def randomPlay(self):
        if self.playerNo == 1:
            ori_pos = deepcopy(self.my_pos)
            my_pos = deepcopy(self.my_pos)
            adv_pos = deepcopy(self.adv_pos)
        else:
            ori_pos = deepcopy(self.adv_pos)
            my_pos = deepcopy(self.adv_pos)
            adv_pos = deepcopy(self.my_pos)

        steps = np.random.randint(0, self.max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        new_board = set_barrier(r, c, dir, deepcopy(self.chess_board))
        if self.playerNo == 1:
            newNode = State(self.max_step)
            newNode.setBoard(new_board)
            newNode.setMypos(my_pos)
            newNode.setAdvpos(adv_pos)
            newNode.setPlayerNo(2)
        else:
            newNode = State(self.max_step)
            newNode.setBoard(new_board)
            newNode.setMypos(adv_pos)
            newNode.setAdvpos(my_pos)
            newNode.setPlayerNo(1)
        return newNode

###################################################################################### student agent class

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
        self.tree = None
        self.sequence = []


    def minimax(self, node, isMaxTurn):

        if node.checkStatus()[0]:
            node.setScore(node.checkStatus()[1])
            return node.checkStatus()[1]
        else:
            expandNode(node)

        scores = []
        for childNode in node.getLeaf():
            scores.append(self.minimax(childNode, not isMaxTurn))

        node.setScore(max(scores))
        return max(scores) if isMaxTurn else min(scores)

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

        if self.stepCount == 1:
            root = State(max_step)
            root.setBoard(chess_board)
            root.setMypos(my_pos)
            root.setAdvpos(adv_pos)
            root.setPlayerNo(1)  # 1 means my turn, 2 means opponent's turn
            self.minimax(root, True)
            self.tree = root
            self.sequence.append(root)

        next_step = findBestNodeWithScore(self.sequence[-1])
        self.sequence.append(next_step)

        i, j = next_step.getMypos()
        for dir in range(4):
            if next_step.getBoard()[i, j, dir] != chess_board[i, j, dir]:
                break

        return next_step.getMypos(), dir

