# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util, operator
from util import nearestPoint
from game import Directions
from collections import defaultdict
import game

POWERCAPSULETIME = 120


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='TopAgent', second='BottomAgent'):
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
class MCTagent(CaptureAgent):
    # Use a list to store the position of the agent. A timer to store the left power time
    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)
        self.agentPosProb = [None] * 4
        self.boundary = []

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
        # Get the Size of the Map
        width, height = gameState.data.layout.width, gameState.data.layout.height
        # Get the color of the team
        if self.red:
            CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
        else:
            CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())
        # Get all the move position in the map
        self.legalMovePositions = [p for p in gameState.getWalls().asList(False)]
        self.walls = list(gameState.getWalls())

        # possible is used to guess the position of the enemy
        global possible
        possible = [util.Counter() for i in range(gameState.getNumAgents())]
        # possible start from the initial position
        for i, value in enumerate(possible):
            if i in self.getOpponents(gameState):
                # Get the initial position of the agent,set the possibility to 1
                possible[i][gameState.getInitialAgentPosition(i)] = 1.0

        # Get the boundary position
        for i in range(height):
            if not self.walls[width // 2][i]:
                self.boundary.append((width//2, i))

        # At the start of the game, two agent go direct to the center
        self.goToCenter(gameState)

    # Determine the time of the ghost scared
    def ScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    # Guess the possible movement of the enemy
    def getGuessedPosition(self, p):
        possibleAction = [(p[0] - 1, p[1]), (p[0], p[1] - 1), (p[0] + 1, p[1]), (p[0], p[1] + 1)]
        dist = util.Counter()
        for act in possibleAction:
            if act in self.legalMovePositions:
                dist[act] = 1
        return dist

    #Get Enemy Position
    def getEnemyPos(self, gameState):
        Pos = []
        for enemy in self.getOpponents(gameState):
            p = gameState.getAgentPosition(enemy)
            if p is not None:
                Pos.append((enemy, p))
        return Pos

    # Get Enemy Distance
    def getDisToEnemy(self, gameState):
        pos = self.getEnemyPos(gameState)
        minDist = None
        if len(pos) > 0:
            dist = []
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist.append(self.getMazeDistance(p, myPos))
            minDist = min(dist)
        return minDist

    # Guess the position of the enemy right now.
    def getPositionNow(self, gameState):
        for enemy, value in enumerate(possible):
            if enemy in self.getOpponents(gameState):
                newBeliefs = util.Counter()
                # get the enemy position if we can see or just guess if not
                pos = gameState.getAgentPosition(enemy)
                if pos is not None:
                    newBeliefs[pos] = 1.0
                else:
                    for p in value:
                        if p in self.legalMovePositions and value[p] > 0:
                            # get all possible movement
                            newPosDist = self.getGuessedPosition(p)
                            for key in newPosDist.keys():
                                newBeliefs[key] += value[p] * newPosDist[key]
                    if len(newBeliefs) == 0:
                        oldState = self.getPreviousObservation()
                        # enemy is eaten
                        if oldState is not None and oldState.getAgentPosition(enemy) is not None:
                            newBeliefs[oldState.getInitialAgentPosition(enemy)] = 1.0
                        else:
                            for p in self.legalMovePositions:
                                newBeliefs[p] = 1.0
                possible[enemy] = newBeliefs
    # Calculate the distance to Partner
    def getDistToPartner(self, gameState):
        # distanceToAgent = None
        agentsList = self.agentsOnTeam
        if self.index == agentsList[0]:
            otherAgentIndex = agentsList[1]
            distanceToAgent = None
        else:
            otherAgentIndex = agentsList[0]
            myPos = gameState.getAgentState(self.index).getPosition()
            otherPos = gameState.getAgentState(otherAgentIndex).getPosition()
            distanceToAgent = self.getMazeDistance(myPos, otherPos)
            if distanceToAgent == 0:
                distanceToAgent = 0.5
        return distanceToAgent

    # Get the distance to Home side
    def getDistanceToHome(self, gameState):
        dis = []
        myPos = gameState.getAgentPosition(self.index)
        for p in self.boundary:
            dis.append(self.getMazeDistance(myPos, p))
        return min(dis)

    def observe(self, enemy, noiseDistance, gameState):
        myPos = gameState.getAgentPosition(self.index)
        # Get the probability of the enemy
        for p in self.legalMovePositions:
            tureDis = util.manhattanDistance(myPos, p)
            prob = gameState.getDistanceProb(tureDis, noiseDistance)
            possible[enemy][p] *= prob

    def evaluate(self, gameState, action, evaluateType):
        """
        Computes a linear combination of features and feature weights
        """
        if evaluateType == 'attack':
            features = self.getFeaturesAttack(gameState, action)
            weights = self.getWeightsAttack()
        elif evaluateType == 'defend':
            features = self.getFeaturesDefend(gameState, action)
            weights = self.getWeightsDefend()
        elif evaluateType == 'start':
            features = self.getFeaturesStart(gameState, action)
            weights = self.getWeightsStart()
        elif evaluateType == 'hunt':
            features = self.getFeaturesHunt(gameState, action)
            weights = self.getWeightsHunt()
        return features * weights

    # Monte Carlo Simulation
    def simulation(self, depth, gameState, decay, evaluateType):
        if depth == 0:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            for action in actions:
                newState = gameState.generateSuccessor(self.index, action)
                simuResult.append(self.evaluate(newState, Directions.STOP, evaluateType))
            return max(simuResult)
        else:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            for action in actions:
                newState = gameState.generateSuccessor(self.index, action)
                simuResult.append(self.evaluate(newState, Directions.STOP, evaluateType)
                                  + decay * self.simulation(depth - 1, newState, decay, evaluateType))
            return max(simuResult)

    def renewPossiblePos(self, gameState):
        opponents = self.getOpponents(gameState)
        noiseDis = gameState.getAgentDistances()
        for a in opponents:
            self.observe(a, noiseDis[a], gameState)
        # Nomorlize the probability and get the mostlikely position
        for agent in opponents:
            possible[agent].normalize()
            self.agentPosProb[agent] = max(possible[agent].items(), key=lambda x: x[1])[0]
        # Guess the next step
        self.getPositionNow(gameState)

    def chooseAction(self, gameState):
        self.renewPossiblePos(gameState)
        # Choose Tactics
        # Set Default Mode
        evaluateType = 'attack'
        opponents = self.getOpponents(gameState)
        agentPos = gameState.getAgentPosition(self.index)

        if self.atCenter is False:
            evaluateType = 'start'

        if agentPos == self.center and self.atCenter is False:
            self.atCenter = True
            evaluateType = 'attack'
        # If enemy is in our side, search it
        for agent in opponents:
            if gameState.getAgentState(agent).isPacman:
                evaluateType = 'hunt'

        ememyPos = self.getEnemyPos(gameState)
        if len(ememyPos) > 0:
            for enemy, pos in ememyPos:
                if self.getMazeDistance(agentPos, pos) < 5 and not gameState.getAgentState(self.index).isPacman:
                    evaluateType = 'defend'
                    break
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        simuResult = []
        for action in actions:
            value = self.simulation(2, gameState.generateSuccessor(self.index, action), 0.7, evaluateType)
            simuResult.append(value)
        maxResult = max(simuResult)
        bestActions = [a for a, v in zip(actions, simuResult) if v == maxResult]
        chosenAction = random.choice(bestActions)
        return chosenAction

    def getFeaturesHunt(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        # Get Own Position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Get the invaders
        opponents = self.getOpponents(gameState)
        invaders = [enemy for enemy in opponents if gameState.getAgentState(enemy).isPacman]
        features['numofInvaders'] = len(invaders)

        # for each invader, get the distance
        enemyDis = []
        for enemy in invaders:
            enemyP = self.agentPosProb[enemy]
            enemyDis.append(self.getMazeDistance(myPos, enemyP))
        if len(enemyDis) > 0:
            features['enemyDistance'] = min(enemyDis)
        else:
            features['enemyDistance'] = 0

        # Compute the distance to Partner
        disToPartner = None
        if successor.getAgentState(self.index).isPacman:
            disToPartner = self.getDistToPartner(successor)
        if disToPartner is not None:
            features['disToPartner'] = 1.0 / disToPartner

        # Danger Index
        powerTime = min([successor.getAgentState(i).scaredTimer for i in self.getOpponents(gameState)])
        disToEnemy = self.getDisToEnemy(successor)
        if disToEnemy is not None:
            if disToEnemy <= 2 and powerTime > 2:
                features['danger'] = 4 / disToEnemy
            elif disToEnemy <= 4 and powerTime > 4:
                features['danger'] = 1
            else:
                features['danger'] = 0

        # Compute the distance to Home
        # if not successor.getAgentState(self.index).isPacman:
        #     disToHome = self.getDistanceToHome(successor)
        #     features['disToHome'] = disToHome

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def getWeightsHunt(self):
        return {'numofInvaders': -100, 'enemyDistance': -10, 'stop': -5000,
                'reverse': -5000, 'disToPartner': -2500, 'danger': -1000} #, 'disToHome': -100}

    def getFeaturesAttack(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        # Get Own Position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Get Food
        foodList = self.getFood(successor).asList()

        # Get the score right now
        features['successScore'] = self.getScore(successor)

        #Distance to nearest food
        if len(foodList) > 0:
            features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

        #Pickup Food
        if len(foodList) > 0:
            features['pickupFood'] = -len(foodList) + 100 * self.getScore(successor)

        # Compute distance to
        powerTime = min([successor.getAgentState(i).scaredTimer for i in self.getOpponents(gameState)])
        disToEnemy = self.getDisToEnemy(successor)
        if disToEnemy is not None:
            if disToEnemy <= 2 and powerTime < 2:
                features['danger'] = 4 / disToEnemy
            elif disToEnemy <= 4 and powerTime < 4:
                features['danger'] = 1
            else:
                features['danger'] = 0

        # Compute distance to capsule
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            features['pickupCapsule'] = -len(capsules)
            features['disToCapsule'] = 1.0 / min([self.getMazeDistance(myPos, p) for p in capsules])

        # Compute holding food heuristic
        foodNum = successor.getAgentState(self.index).numCarrying

        features['foodNum'] = foodNum
        features['holdFood'] = foodNum * self.getDistanceToHome(successor)

        # If pick up a capsule, set PowerTime
        if powerTime > 0:
            features['isPowered'] = powerTime / POWERCAPSULETIME
            features['pickupFood'] = 100 * features['pickupFood']
        else:
            features['isPowered'] = 0.0

        # Compute distance to partner
        if successor.getAgentState(self.index).isPacman:
            disToPartner = self.getDistToPartner(successor)
            if disToPartner is not None:
                features['disToPartner'] = 1.0 / disToPartner

        # Compute the dead end
        actions = successor.getLegalActions(self.index)
        if len(actions) < 2:
            features['deadEnd'] = 1.0
        else:
            features['deadEnd'] = 0.0

        # Stop heuristic
        if action == Directions.STOP:
            features['stop'] = 1.0
        else:
            features['stop'] = 0.0

        return features

    def getWeightsAttack(self):
        return {'successorScore': 800, 'distanceToFood': -10, 'danger': -1000,
                'pickupFood': 4000, 'disToCapsule': 700, 'stop': -1000, 'deadEnd': -200,
                'isPowered': 5000000, 'foodNum': 100, 'holdFood': -20,
                'disToPartner': -6000, 'pickupCapsule': 5000}

    def getFeaturesDefend(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        # Get Own Position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # List the invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

        # Get the number of invader we can see
        features['numofInvaders'] = len(invaders)

        # Get the distance of the invader
        if len(invaders) > 0:
            features['disToInvaders'] = min([self.getMazeDistance(invader.getPosition(), myPos) for invader in invaders])

        # Compute distance to invader if get scared
        if self.ScaredTimer(successor) > 0:
            disToEnemy = self.getDisToEnemy(successor)
            if disToEnemy <= 5:
                features['danger'] = 1
            elif disToEnemy <= 1:
                features['danger'] = -1
        else:
            features['danger'] = 0
        ##################
        ## Improve Part ##
        ##################
        # Compute Distance to Partner
        disToPartner = None
        if successor.getAgentState(self.index).isPacman:
            disToPartner = self.getDistToPartner(successor)
        if disToPartner is not None:
            features['disToPartner'] = 1.0 / disToPartner

        # Compute Distance to Capsule Need Protect #

        # Compute the end
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeightsDefend(self):
        return {'numofInvaders': -10000, 'disToInvaders': -500, 'stop': -5000,
                'reverse': -200, 'danger': 3000, 'disToPartner': -4000}

    def getFeaturesStart(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        # Get Own Position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute Distance to Center
        dist = self.getMazeDistance(myPos, self.center)
        features['distToCenter'] = dist
        if myPos == self.center:
            features['atCenter'] = 1
        return features

    def getWeightsStart(self):
        return {'distToCenter': -1, 'atCenter': 1000}

#
class TopAgent(MCTagent):

    def goToCenter(self, gameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width // 2
        y = gameState.getWalls().height // 2
        # 0 to x-1 and x to width
        if self.red:
            x = x - 1
        # Set where the centre is
        self.center = (x, y)
        maxHeight = gameState.getWalls().height

        # Look for locations to move to that are not walls (favor top positions)
        for i in range(maxHeight - y):
            if not gameState.hasWall(x, y):
                locations.append((x, y))
            y = y + 1

        myPos = gameState.getAgentState(self.index).getPosition()
        minDist = float('inf')
        minPos = None

        # Find shortest distance to centre
        for location in locations:
            dist = self.getMazeDistance(myPos, location)
            if dist <= minDist:
                minDist = dist
                minPos = location

        self.center = minPos


# Agent that has a bias to moving around the bottom of the board
class BottomAgent(MCTagent):

    def goToCenter(self, gameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width // 2
        y = gameState.getWalls().height // 2
        # 0 to x-1 and x to width
        if self.red:
            x = x - 1
        # Set where the centre is
        self.center = (x, y)

        # Look for locations to move to that are not walls (favor bot positions)
        for i in range(y):
            if not gameState.hasWall(x, y):
                locations.append((x, y))
            y = y - 1

        myPos = gameState.getAgentState(self.index).getPosition()
        minDist = float('inf')
        minPos = None

        # Find shortest distance to centre
        for location in locations:
            dist = self.getMazeDistance(myPos, location)
            if dist <= minDist:
                minDist = dist
                minPos = location

        self.center = minPos








































