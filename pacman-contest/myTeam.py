# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'AttackAgent', second = 'DefendAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

  # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

'''
The GeneralAgent is the parent class of AttackAgent and DefendAgent. Several fundamental variables and methods are
declared in this class.
'''

class GeneralAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        '''
        Your initialization code goes here, if you need any.
        '''
        self.start = gameState.getAgentPosition(self.index)
        self.maxHeight = gameState.data.layout.height
        self.maxWidth = gameState.data.layout.width
        self.midLine = int(self.maxWidth / 2)
        self.lastFoodEaten = None

        preProcessor = preProcessing(self, gameState)
        foodSets = preProcessor.evaluateFood(gameState)
        self.easyFoods = foodSets[0]
        self.hardFoods = foodSets[1]

    # Return a move based on the action taken
    def getMove(self, gameState, action):
        """
        Finds the next successor (Game state object)
        """
        move = gameState.generateSuccessor(self.index, action)
        position = move.getAgentState(self.index).getPosition()

        if position == util.nearestPoint(position):
            return move
        else:
            return move.generateSuccessor(self.index, action)

    def getWeights(self, gameState, action):
        return {'Score': 1.0}

    def getFeatures(self, gameState, action):
        features = util.Counter()
        move = self.getMove(gameState, action)
        features['moveScore'] = self.getScore(move)

        return features

    # Compute the corresponding value of each action
    def compute(self, gameState, action):
        weights = self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)
        result = features * weights

        return result

    # Return the distance to the nearest capsule
    def distanceToCapsule(self, gameState):
        position = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        distances = []

        for capsule in capsules:
            distances.append(self.getMazeDistance(position, capsule))

        return min(distances)

    # Return the distance to the nearest home boundary
    def distanceHome(self, gameState):
        position = gameState.getAgentState(self.index).getPosition()
        validBoundaries = self.getValidBoundaries(gameState)
        distances = []

        for validBoundary in validBoundaries:
            distances.append(self.getMazeDistance(validBoundary, position))

        return min(distances)

    # Return the distance to the nearest detected invader
    def distanceToNearestInvader(self, gameState):
        position = gameState.getAgentState(self.index).getPosition()
        opponentsIds = self.getOpponents(gameState)
        opponents = []
        invadersPositions = []
        distances = []

        for opponentsId in opponentsIds:
            opponents.append(gameState.getAgentState(opponentsId))

        for opponent in opponents:

            if (opponent.getPosition() is not None) and (opponent.isPacman):
                invadersPositions.append(opponent.getPosition())

        if len(invadersPositions) >= 1:

            for invaderPosition in invadersPositions:
                distances.append(self.getMazeDistance(invaderPosition, position))
            return min(distances)
        else:
            return None

    # Return a list of boundaries that are not walls
    def getValidBoundaries(self, gameState):

        validBoundaries = []
        if self.red:
            for y in range(self.maxHeight):
                if not gameState.hasWall(self.midLine - 1, y):
                    validBoundaries.append((self.midLine - 1, y))
        else:
            for y in range(self.maxHeight):
                if not gameState.hasWall(self.midLine + 1, y):
                    validBoundaries.append((self.midLine + 1, y))

        return validBoundaries

    # Return the last food eaten by opponents
    def getLastFoodEaten(self, gameState):

        if len(self.observationHistory) >= 2:
            previousState = self.getPreviousObservation()
            previousFoods = self.getFoodYouAreDefending(previousState).asList()
            currentFoods = self.getFoodYouAreDefending(gameState).asList()

            if len(previousFoods) != len(currentFoods):
                for food in previousFoods:
                    if food not in currentFoods:
                        self.lastFoodEaten = food

    # Return the distance to the nearest detected ghost and the sate of ghost when attacking
    def getNearestGhost(self, gameState):

        position = gameState.getAgentState(self.index).getPosition()
        opponentsIds = self.getOpponents(gameState)
        opponents = []
        ghosts = []

        for opponentId in opponentsIds:
            opponents.append(gameState.getAgentState(opponentId))

        for opponent in opponents:

            if (opponent.getPosition() is not None) and (not opponent.isPacman):
                ghosts.append(opponent)

        if len(ghosts) >= 1:
            nearestGhost = [10000, None]

            for ghost in ghosts:
                distance = self.getMazeDistance(ghost.getPosition(), position)

                if distance < nearestGhost[0]:
                    nearestGhost = [distance, ghost]
            return nearestGhost
        else:
            return None

    # Return the remaining scared time of opponents
    def ghostsScaredTimer(self, gameState):
        opponentsIds = self.getOpponents(gameState)

        for opponentId in opponentsIds:
            timer = gameState.getAgentState(opponentId).scaredTimer

            if timer >= 2:
                return timer

        return 0

    def nullHeuristic(self, state, problem=None):

        return 0

    # The primary heuristic used in most cases
    def avoidGhostHeuristic(self, state, gameState):
        heuristic = 0

        if self.getNearestGhost(gameState) is not None:
            opponentsIds = self.getOpponents(gameState)
            opponents = []
            ghosts = []
            for opponentId in opponentsIds:
                opponents.append(gameState.getAgentState(opponentId))

            for opponent in opponents:
                if (not opponent.isPacman) and (opponent.scaredTimer <= 1) and (opponent.getPosition() is not None):
                    ghosts.append(opponent)

            if len(ghosts) >= 1:
                ghostsDistances = []
                for ghost in ghosts:
                    ghostsDistances.append(self.getMazeDistance(ghost.getPosition(), state))

                minDistance = min(ghostsDistances)
                '''TODO: tweak minDistance limit'''
                if minDistance < 3:
                    heuristic = pow((5 - minDistance), 5)

        return heuristic

    # Used by scared DefendAgent in defending mode
    def avoidPacmanHeuristic(self, state, gameState):
        heuristic = 0
        if (self.distanceToNearestInvader(gameState) is not None) and (gameState.getAgentState(self.index).scaredTimer >= 1):
            opponentsIds = self.getOpponents(gameState)
            opponents = []
            pacmans = []
            for opponentId in opponentsIds:
                opponents.append(gameState.getAgentState(opponentId))

            for opponent in opponents:
                if (opponent.isPacman) and (opponent.getPosition() is not None):
                    pacmans.append(opponent)

            if len(pacmans) >= 1:
                pacmenDistances = []
                for pacman in pacmans:
                    pacmenDistances.append(self.getMazeDistance(pacman.getPosition(), state))

                minDistance = min(pacmenDistances)
                '''TODO: tweak minDistance limit'''
                if minDistance < 3:
                    heuristic = pow((5 - minDistance), 5)

        return heuristic

    # A-Star Search algorithm, the core foundation of the technique
    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
        startState = problem.getStartState()
        hValue = heuristic(startState, gameState)
        gValue = 0
        fValue = hValue + gValue
        startNode = {'state': startState, 'action': None, 'cost': gValue, 'parent': None}
        openList = util.PriorityQueue()
        openList.push(startNode, fValue)
        closedList = []
        currentNode = {}

        while not openList.isEmpty():
            currentNode = openList.pop()

            if currentNode['state'] not in closedList:
                closedList.append(currentNode['state'])

                if problem.isGoalState(currentNode['state']):
                    break

                successors = problem.getSuccessors(currentNode['state'])

                for successor in successors:
                    gvalue = currentNode['cost'] + successor[2]
                    fvalue = gvalue + heuristic(successor[0], gameState)
                    openList.push({'state': successor[0], 'action': successor[1], 'cost': gvalue,
                                   'parent': currentNode}, fvalue)

        actionList = []

        while currentNode['parent'] is not None:
            actionList.append(currentNode['action'])
            currentNode = currentNode['parent']

        actionList.reverse()
        return actionList


# The first agent which is focusing on attacking
class AttackAgent(GeneralAgent):

    def chooseAction(self, gameState):
        updatedEasyFoods = []
        updatedHardFoods = []

        # Update the classified food sets according to the current game state
        for food in self.getFood(gameState).asList():
            if food in self.easyFoods:
                updatedEasyFoods.append(food)
            elif food in self.hardFoods:
                updatedHardFoods.append(food)

        self.easyFoods = updatedEasyFoods
        self.hardFoods = updatedHardFoods

        # The decision tree to deal with different scenarios
        if gameState.getAgentState(self.index).numCarrying < 1 and len(self.getFood(gameState).asList()) < 1:
            return 'Stop'

        if (len(self.easyFoods) == 0) and (len(self.getCapsules(gameState)) >= 1) and (self.ghostsScaredTimer(gameState) <= 9):
            problem = CapsuleSearchProblem(gameState, self, self.index)
            return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        if (gameState.getAgentState(self.index).numCarrying == 0) and (len(self.easyFoods) >= 1):
            problem = EasyFoodSearchProblem(gameState, self, self.index)
            return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        if (gameState.getAgentState(self.index).numCarrying == 0) and (len(self.easyFoods) < 1):
            problem = FoodSearchProblem(gameState, self, self.index)
            return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        if (self.getNearestGhost(gameState) is not None) and (self.getNearestGhost(gameState)[0] <= 5) and (self.getNearestGhost(gameState)[1].scaredTimer <= 4):
            problem = EscapeProblem(gameState, self, self.index)
            if len(self.aStarSearch(problem, self.avoidGhostHeuristic)) == 0:
                return 'Stop'
            else:
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        if self.ghostsScaredTimer(gameState) is not None:
            if self.ghostsScaredTimer(gameState) >= 20 and len(self.hardFoods) >= 1:
                problem = HardFoodSearchProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        '''TODO: tweak numCarrying below'''
        if (len(self.getFood(gameState).asList()) <= 2) or ((gameState.data.timeleft/4) < (self.distanceHome(gameState) + 15)) or (gameState.getAgentState(self.index).numCarrying >= 16):
            problem = ReturnHomeProblem(gameState, self, self.index)
            if len(self.aStarSearch(problem, self.avoidGhostHeuristic)) == 0:
                return 'Stop'
            else:
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        problem = FoodSearchProblem(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]


# The second agent which is focusing on defending in most cases.
class DefendAgent(GeneralAgent):

    def chooseAction(self, gameState):
        self.getLastFoodEaten(gameState)
        legalActions = gameState.getLegalActions(self.index)

        opponentsIds = self.getOpponents(gameState)
        opponents = []
        invaders = []
        detectedInvaders = []

        for opponentId in opponentsIds:
            opponents.append(gameState.getAgentState(opponentId))

        for opponent in opponents:
            if opponent.isPacman:
                invaders.append(opponent)

        for invader in invaders:
            if invader.getPosition() is not None:
                detectedInvaders.append(invader)

        if (len(invaders) < 1) or (len(detectedInvaders) >= 1) or (gameState.getAgentPosition(self.index) == self.lastFoodEaten):
            self.lastFoodEaten = None

        # Focus on defense when opponents are attacking
        if len(invaders) > 0:
            if (len(detectedInvaders) < 1) and (self.lastFoodEaten is None) and (gameState.getAgentState(self.index).scaredTimer <= 1):
                problem = InvaderSearchProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

            if (len(detectedInvaders) < 1) and (self.lastFoodEaten is not None) and (gameState.getAgentState(self.index).scaredTimer <= 1):
                problem = LastFoodEatenSearchProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

            if (len(detectedInvaders) < 1) and (self.lastFoodEaten is not None) and (gameState.getAgentState(self.index).scaredTimer > 1):
                problem = LastFoodEatenSearchProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidPacmanHeuristic)[0]
            '''TODO: tweak timer below'''
            # if (len(detectedInvaders) >= 1) and ((gameState.getAgentState(self.index).scaredTimer <= 1) or (gameState.getAgentState(self.index).scaredTimer > 1) and (self.getNearestInvader(gameState, detectedInvaders) >= 3)):
            if (len(detectedInvaders) >= 1) and (gameState.getAgentState(self.index).scaredTimer <= 1):
                problem = InvaderHuntProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

            if (len(detectedInvaders) >= 1) and (gameState.getAgentState(self.index).scaredTimer > 1) and (self.getNearestInvader(gameState, detectedInvaders) >= 3):
                problem = InvaderHuntProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidPacmanHeuristic)[0]
        # Switch to a limited attacking mode when no opponent is attacking
        else:
            if (gameState.getAgentState(self.index).numCarrying <= 2) and (len(self.getFood(gameState).asList()) >= 1) and (not (self.getNearestGhost(gameState) is not None and self.getNearestGhost(gameState)[0] <= 3 and self.getNearestGhost(gameState)[1].scaredTimer <= 1)):
                problem = FoodSearchProblem(gameState, self, self.index)
                return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]
            else:
                problem = ReturnHomeProblem(gameState, self, self.index)
                if len(self.aStarSearch(problem, self.avoidGhostHeuristic)) == 0:
                    return 'Stop'
                else:
                    return self.aStarSearch(problem, gameState, self.avoidGhostHeuristic)[0]

        bestValue = -10000
        bestActions = []

        for action in legalActions:
            value = self.compute(gameState, action)
            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        return random.choice(bestActions)

    # Return the distance to the nearest invader
    def getNearestInvader(self, gameState, detectedInvaders):
        position = gameState.getAgentState(self.index).getPosition()
        distances = []
        for detectedInvader in detectedInvaders:
            distances.append(self.getMazeDistance(position, detectedInvader.getPosition()))

        return min(distances)

    # Overridden methods of parent class to choose action
    def getFeatures(self, gameState, action):
        features = util.Counter()
        move = self.getMove(gameState, action)

        state = move.getAgentState(self.index)
        position = state.getPosition()

        features['dead'] = 0

        if not state.isPacman:
            features['onDefense'] = 1
        else:
            features['onDefense'] = 0

        opponentsIds = self.getOpponents(gameState)
        opponents = []
        detectedInvaders = []

        for opponentId in opponentsIds:
            opponents.append(gameState.getAgentState(opponentId))

        for opponent in opponents:
            if (opponent.getPosition() is not None) and (opponent.isPacman):
                detectedInvaders.append(opponent)

        features['numInvaders'] = len(detectedInvaders)
        if (len(detectedInvaders) >= 1) and (gameState.getAgentState(self.index).scaredTimer >= 1):
            distances = []
            for detectedInvader in detectedInvaders:
                distances.append(self.getMazeDistance(position, detectedInvader.getPosition()))

            features['invaderDistance'] = -1 / min(distances)

        if action == Directions.STOP:
            features['stop'] = 1

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == reverse:
            features['reverse'] = 1
        features['DistToBoundary'] = - self.distanceHome(move)

        return features

    def getWeights(self, gameState, action):
        return {'invaderDistance': 1000, 'onDefense': 200, 'stop': -100, 'reverse': -2, 'DistToBoundary': 1, 'dead': -10000}


# The parent class of different classes for different scenarios
# Templates of all the problem classes are referenced from assignment 1
class PositionSearchProblem:

    def __init__(self, gameState, agent, agentIndex = 0,costFn = lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = game.Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


# To search the capsule for escaping or further attacking
class CapsuleSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.capsules = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state in self.capsules


# To search food in common situations
class FoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.foods = agent.getFood(gameState).asList()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state in self.foods


# To search food with low risks if there is a threat
class EasyFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.easyFoods = agent.easyFoods
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state in self.easyFoods


# To search food that hard to escape from when it seems safe
class HardFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.hardFoods = agent.hardFoods
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state in self.hardFoods


# Get a capsule or go home if being chased
class EscapeProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex=0):
        self.capsules = agent.getCapsules(gameState)
        self.homeBoundary = agent.getValidBoundaries(gameState)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return (state in self.homeBoundary) or (state in self.capsules)

    def getStartState(self):
        return self.startState


# Return food to score
class ReturnHomeProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex=0):
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0
        self.homeBoundary = agent.getValidBoundaries(gameState)

    def isGoalState(self, state):
        return state in self.homeBoundary

    def getStartState(self):
        return self.startState


# Go to check the last food eaten by the opponent to find invaders
class LastFoodEatenSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.lastFoodEaten = agent.lastFoodEaten
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state == self.lastFoodEaten


# Search for invaders in case no food has been eaten so far
class InvaderSearchProblem(PositionSearchProblem):
    def __init__(self, gameState, agent, agentIndex = 0):
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.agent = agent
        self.gameState = gameState
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return self.agent.distanceToNearestInvader(self.gameState) is not None


# Try to kill the detected invaders
class InvaderHuntProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

        opponentsIds = agent.getOpponents(gameState)
        opponents = []
        self.invaders = []
        self.invadersPositions = []

        for opponentId in opponentsIds:
            opponents.append(gameState.getAgentState(opponentId))

        for opponent in opponents:
            if (opponent.getPosition() is not None) and opponent.isPacman:
                self.invaders.append(opponent)

        if len(self.invaders) >= 1:
            for invader in self.invaders:
                self.invadersPositions.append(invader.getPosition())

    def isGoalState(self, state):
        return state in self.invadersPositions


# The preprocessing method to get the environment of each food and classify all the food into to sets
# Used in the initial preparation stage
class preProcessing:

    def __init__(self, agent, gameState):
        self.foods = agent.getFood(gameState).asList()
        self.walls = gameState.getWalls()
        self.homeBoundaries = agent.getValidBoundaries(gameState)
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width

    # Return the surroundings of each food which are not walls
    def getFoodsEnv(self, gameState):
        foodEnv = []

        for food in self.foods:
            neighbours = [(food[0], food[1] + 1), (food[0], food[1] - 1), (food[0] + 1, food[1]), (food[0] - 1, food[1])]
            notWallNeighbours = []

            for neighbour in neighbours:
                if not gameState.hasWall(neighbour[0], neighbour[1]):
                    notWallNeighbours.append(neighbour)

            if len(notWallNeighbours) >= 2:
                foodEnv.append((food, notWallNeighbours))

        return foodEnv

    # Return the list of food that is easy for pacman to escape after eating
    def getEasyFoods(self, foodEnv):
        easyFoods = []

        for food in foodEnv:
            routes= self.getRoutesNum(food)

            if routes > 1:
                easyFoods.append(food[0])

        return easyFoods

    # Return the list of food that need to be cautious when eating
    def getHardFoods(self, easyFoods):
        hardFoods = []

        for food in self.foods:

            if food not in easyFoods:
                hardFoods.append(food)

        return hardFoods

    def getSuccessors(self, state):
        successors = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:
                successors.append((nextx, nexty))

        return successors

    def isGoalState(self, state):

        return state in self.homeBoundaries

    # Return the number of different routes from the food to home boundary
    def getRoutesNum(self, food):
        routeNum = 0
        foodLocation = food[0]
        foodNeighbours = food[1]

        for neighbour in foodNeighbours:
            closedList = [foodLocation]

            if self.breadthFirstSearch(neighbour, closedList):
                routeNum += 1

        return routeNum

    def breadthFirstSearch(self, neighbour, closedList):
        openList = util.Queue()
        openList.push(neighbour)

        while not openList.isEmpty():
            currentState = openList.pop()
            closedList.append(currentState)

            if self.isGoalState(currentState):
                return True

            successors = self.getSuccessors(currentState)
            for successor in successors:
                if successor not in closedList:
                    closedList.append(successor)
                    openList.push(successor)

    # Classifier of food
    def evaluateFood(self, gameState):
        foodEnv = self.getFoodsEnv(gameState)
        easyFoods = self.getEasyFoods(foodEnv)
        hardFoods = self.getHardFoods(easyFoods)

        return easyFoods, hardFoods
