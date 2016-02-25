from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *


__author__ = 'sshadmin'

import math
import random


Inputs = 1
Outputs = 1

Population = 150
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

TimeoutConstant = 20

MaxNodes = 1000000


ButtonNames = [
                "JUMP"
        ]

def newInnovation():
        pool["innovation"] = pool["innovation"] + 1
        return pool["innovation"]

def sigmoid(x):
    return 2/(1+math.exp(-4.9*x))-1

def newNeuron():
    neuron = dict()
    neuron["incoming"]=[]
    neuron["value"] = 0.0
    return neuron

def newGene():
    gene= dict()
    gene["into"] = 0
    gene["out"] = 0
    gene["weight"] = 0.0
    gene["enabled"] = True
    gene["innovation"] = 0
    return gene

def copyGene(gene):
    gene2=newGene()
    gene2["into"] = gene["into"]
    gene2["out"] = gene["out"]
    gene2["weight"] = gene["weight"]
    gene2["enabled"] = gene["enabled"]
    gene2["innovation"] = gene["innovation"]
    return gene2

def basicGenome():
    genome = newGenome()
    innovation = 1
    genome["maxneuron"] = Inputs
    mutate(genome)
    return genome

def newGenome():
    genome = dict()
    genome["genes"] = []
    genome["fitness"] = 0
    genome["adjustedFitness"] = 0
    genome["network"] = dict()
    genome["maxneuron"] = 0
    genome["globalRank"] = 0
    genome["mutationRates"] = dict()
    genome["mutationRates"]["connections"] = MutateConnectionsChance
    genome["mutationRates"]["link"] = LinkMutationChance
    genome["mutationRates"]["bias"] = BiasMutationChance
    genome["mutationRates"]["node"] = NodeMutationChance
    genome["mutationRates"]["enable"] = EnableMutationChance
    genome["mutationRates"]["disable"] = DisableMutationChance
    genome["mutationRates"]["step"] = StepSize
    return genome

def copyGenome(genome):
    genome2 = newGenome()
    for g in range(len(genome["genes"])):
        genome2["genes"].append(copyGene(genome["genes"][g]))
    genome2["maxneuron"] = genome["maxneuron"]
    genome2["mutationRates"]["connections"] = genome["mutationRates"]["connections"]
    genome2["mutationRates"]["link"] = genome["mutationRates"]["link"]
    genome2["mutationRates"]["bias"] = genome["mutationRates"]["bias"]
    genome2["mutationRates"]["node"] = genome["mutationRates"]["node"]
    genome2["mutationRates"]["enable"] = genome["mutationRates"]["enable"]
    genome2["mutationRates"]["disable"] = genome["mutationRates"]["disable"]
    return genome2

def newPool():
    pool = dict()
    pool["species"] = []
    pool["generation"] = 0
    pool["innovation"] = Outputs
    pool["currentSpecies"] = 0
    pool["currentGenome"] = 0
    pool["currentFrame"] = 0
    pool["maxFitness"] = 0
    return pool

def newSpecies():
    species = dict()
    species["topFitness"] = 0
    species["staleness"] = 0
    species["genomes"] = []
    species["averageFitness"] = 0
    return species

def generateNetwork(genome):
    network = dict()
    network["neurons"] = dict()

    for j in range(Inputs):
        network["neurons"][j] = newNeuron()
    for o in range(Outputs):
        network["neurons"][MaxNodes+o] = newNeuron()
        genome["genes"].sort(key = lambda x : x["out"])
    for j in range(len(genome["genes"])):
        gene = genome["genes"][j]
        if gene["enabled"]:
            if gene["out"] not in network["neurons"]:
                    network["neurons"][gene["out"]] = newNeuron()
            neuron = network["neurons"][gene["out"]]
            neuron["incoming"].append(gene)

            if gene["into"] not in network["neurons"]:
                    network["neurons"][gene["into"]] = newNeuron()
    genome["network"] = network


def evaluateNetwork(network, inputs):
        # inputs.append(1)
        if len(inputs) != Inputs:
                print "Incorrect number of neural network inputs."
                return dict()
        for i in range(Inputs):
                network["neurons"][i]["value"] = inputs[i]

        for _,neuron in network["neurons"].iteritems():
                sum = 0
                for j in range(len(neuron["incoming"])):
                        incoming = neuron["incoming"][j]
                        other = network["neurons"][incoming["into"]]
                        sum = sum + incoming["weight"] * other["value"]
                if len(neuron["incoming"]) >0:
                    neuron["value"] = sigmoid(sum)

        outputs = []
        for o in range(Outputs):
                button = ButtonNames[o]
                if network["neurons"][MaxNodes+o]["value"] > 0:
                        outputs.append(1)
                else:
                        outputs.append(0)
        return outputs

def crossover(g1, g2):
        #Make sure g1 is the higher fitness genome
        if g2["fitness"] > g1["fitness"]:
                tempg = g1
                g1 = g2
                g2 = tempg

        child = newGenome()

        innovations2 = {}
        for i in range(len(g2["genes"])):
                gene = g2["genes"][i]
                innovations2[gene["innovation"]] = gene
        for i in range(len(g1["genes"])):
                gene1 = g1["genes"][i]
                gene2 = innovations2[gene1["innovation"]] if gene1["innovation"] in innovations2 else None
                if gene2 != None and random.randint(1,2) == 1 and gene2["enabled"]:
                        child["genes"].append(copyGene(gene2))
                else:
                        child["genes"].append(copyGene(gene1))

        child["maxneuron"] = max(g1["maxneuron"],g2["maxneuron"])

        for mutation,rate in g1["mutationRates"].iteritems():
                child["mutationRates"][mutation] = rate
        return child


def randomNeuron(genes, nonInput):
        neurons = {}
        if not nonInput :
            for i in range(0,Inputs):
                        neurons[i] = True
        for o in range(0,Outputs):
                neurons[MaxNodes+o] = True
        for i in range(0,len(genes)):
                if (not nonInput) or genes[i]["into"] > Inputs:
                        neurons[genes[i]["into"]] = True
                if (not nonInput) or genes[i]["out"] > Inputs:
                        neurons[genes[i]["out"]] = True

        count = 0
        for _,_ in neurons.iteritems():
                count = count + 1

        n=random.randint(1,count)

        for k,v in neurons.iteritems():
                n = n-1
                if n == 0:
                        return k
        return 0


def containsLink(genes, link):
        for i in range(len(genes)):
                gene = genes[i]
                if gene["into"] == link["into"] and gene["out"] == link["out"]:
                        return True
        return False

def pointMutate(genome):
        step = genome["mutationRates"]["step"]
        for i in range(len(genome["genes"])):
                gene = genome["genes"][i]
                if random.uniform(0,1) < PerturbChance:
                        gene["weight"] = gene["weight"] + random.uniform(0,1) * step*2 - step
                else:
                        gene["weight"] = random.uniform(0,1)*4-2


def linkMutate(genome, forceBias):
        neuron1 = randomNeuron(genome["genes"], False)
        neuron2 = randomNeuron(genome["genes"], True)

        newLink = newGene()
        if neuron1 <= Inputs and neuron2 <= Inputs:
                #Both input nodes
                return
        if neuron2 <= Inputs:
                #Swap output and input
                temp = neuron1
                neuron1 = neuron2
                neuron2 = temp

        newLink["into"] = neuron1
        newLink["out"] = neuron2
        if forceBias:
                newLink["into"] = Inputs

        if containsLink(genome["genes"], newLink):
                return
        newLink["innovation"] = newInnovation()
        newLink["weight"] = random.uniform(0,1)*4-2
        genome["genes"].append(newLink)

def nodeMutate(genome):

        if len(genome["genes"]) == 0:
                return

        genome["maxneuron"] = genome["maxneuron"] + 1

        gene = genome["genes"][random.randint(0,len(genome["genes"])-1)]
        if not gene["enabled"]:
                return
        gene["enabled"]= False

        gene1 = copyGene(gene)
        gene1["out"]= genome["maxneuron"]
        gene1["weight"] = 1.0
        gene1["innovation"] = newInnovation()
        gene1["enabled"] = True
        genome["genes"].append(gene1)

        gene2 = copyGene(gene)
        gene2["into"] = genome["maxneuron"]
        gene2["innovation"] = newInnovation()
        gene2["enabled"] = True
        genome["genes"].append(gene2)

def enableDisableMutate(genome, enable):
        candidates = []
        for gene in genome["genes"]:
                if gene["enabled"]  != enable:
                        candidates.append(gene)

        if len(candidates) == 0:
                return

        gene = candidates[random.randint(0,len(candidates)-1)]
        gene["enabled"] = not gene["enabled"]

def mutate(genome):
        for mutation,rate in genome["mutationRates"].iteritems():
                if random.randint(1,2) == 1:
                        genome["mutationRates"][mutation] = 0.95*rate
                else:
                        genome["mutationRates"][mutation] = 1.05263*rate

        if random.uniform(0,1) < genome["mutationRates"]["connections"]:
                pointMutate(genome)

        p = genome["mutationRates"]["link"]
        while p > 0 :
                if random.uniform(0,1) < p:
                        linkMutate(genome, False)
                p = p - 1

        p = genome["mutationRates"]["bias"]
        while p > 0 :
                if random.uniform(0,1) < p:
                        linkMutate(genome, True)
                p = p - 1

        p = genome["mutationRates"]["node"]
        while p > 0:
                if random.uniform(0,1) < p:
                        nodeMutate(genome)
                p = p - 1

        p = genome["mutationRates"]["enable"]
        while p > 0:
                if random.uniform(0,1) < p:
                        enableDisableMutate(genome, True)
                p = p - 1

        p = genome["mutationRates"]["disable"]
        while p > 0:
                if random.uniform(0,1) < p :
                        enableDisableMutate(genome, True)
                p = p - 1


def disjoint(genes1, genes2):
        i1 = {}
        for i in range(len(genes1)):
                gene = genes1[i]
                i1[gene["innovation"]] = True

        i2 = {}
        for i in range(len(genes2)):
                gene = genes2[i]
                i2[gene["innovation"]] = True

        disjointGenes = 0
        for i in range(len(genes1)):
                gene = genes1[i]
                if gene["innovation"] not in i2:
                        disjointGenes = disjointGenes+1
        for i in range(len(genes2)):
                gene = genes2[i]
                if gene["innovation"] not in i1:
                        disjointGenes = disjointGenes+1

        n = max(len(genes1), len(genes2))

        return disjointGenes / n


def weights(genes1, genes2):
        i2 = {}
        for i in range(len(genes2)):
                gene = genes2[i]
                i2[gene["innovation"]] = gene

        sum = 0
        coincident = 0
        for i in range(len(genes1)):
                gene = genes1[i]
                if gene["innovation"] in i2:
                        gene2 = i2[gene["innovation"]]
                        sum = sum + abs(gene["weight"] - gene2["weight"])
                        coincident = coincident + 1
        if coincident ==0:
            return float("inf")
        return sum / coincident

def sameSpecies(genome1, genome2):
        dd = DeltaDisjoint*disjoint(genome1["genes"], genome2["genes"])
        dw = DeltaWeights*weights(genome1["genes"], genome2["genes"])
        return dd + dw < DeltaThreshold


# rank genome globally
def rankGlobally():
        global2 = []
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                for g in range(len(species["genomes"])):
                        global2.append(species["genomes"][g])
        global2.sort(key = lambda x : x["fitness"])
        for g in range(len(global2)):
                global2[g]["globalRank"]= g


def calculateAverageFitness(species):
        total = 0
        for g in range(len(species["genomes"])):
                genome = species["genomes"][g]
                total = total + genome["globalRank"]
        species["averageFitness"] = total / len(species["genomes"])

def totalAverageFitness():
        total = 0
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                total = total + species["averageFitness"]
        return total

def cullSpecies(cutToOne):
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                species["genomes"].sort(key = lambda x : x["fitness"], reverse = True)
                remaining = math.ceil(len(species["genomes"])/2.0)
                if cutToOne:
                        remaining = 1
                while len(species["genomes"]) > remaining:
                        species["genomes"].pop()


def breedChild(species):
        child = {}
        if random.uniform(0,1) < CrossoverChance:
                g1 = species["genomes"][random.randint(0,len(species["genomes"])-1)]
                g2 = species["genomes"][random.randint(0,len(species["genomes"])-1)]
                child = crossover(g1, g2)
        else:
                g = species["genomes"][random.randint(0,len(species["genomes"])-1)]
                child = copyGenome(g)
        mutate(child)
        return child

def removeStaleSpecies():
        survived = []
        for s in range(len(pool["species"])):
                species = pool["species"][s]

                species["genomes"].sort(key = lambda x: x["fitness"],reverse=True)

                if species["genomes"][0]["fitness"] > species["topFitness"]:
                        species["topFitness"] = species["genomes"][0]["fitness"]
                        species["staleness"] = 0
                else:
                        species["staleness"] = species["staleness"] + 1
                if species["staleness"] < StaleSpecies or species["topFitness"] >= pool["maxFitness"]:
                        survived.append(species)
        pool["species"] = survived

def removeWeakSpecies():
        survived = []
        sum = totalAverageFitness()
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                breed = math.floor(species["averageFitness"] *1.0/ sum * Population)
                if breed >= 1:
                        survived.append(species)
        pool["species"] = survived


def addToSpecies(child):
        foundSpecies = False
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                if not foundSpecies and sameSpecies(child, species["genomes"][0]):
                        species["genomes"].append(child)
                        foundSpecies = True

        if not foundSpecies :
                childSpecies = newSpecies()
                childSpecies["genomes"].append(child)
                pool["species"].append(childSpecies)


def newGeneration():
        cullSpecies(False) #Cull the bottom half of each species
        rankGlobally()
        removeStaleSpecies()
        rankGlobally()
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                calculateAverageFitness(species)

        removeWeakSpecies()
        sum = totalAverageFitness()
        children = []
        for s in range(len(pool["species"])):
                species = pool["species"][s]
                breed = int(math.floor(species["averageFitness"] *1.0 / sum * Population) - 1)
                for j in range(breed):
                        children.append(breedChild(species))
        cullSpecies(True) # Cull all but the top member of each species
        while len(children) + len(pool["species"]) < Population:
                if len(pool["species"]) ==1:
                    species = pool["species"][0]
                else:
                    species = pool["species"][random.randint(0,len(pool["species"])-1)]
                children.append(breedChild(species))
        for c in range(len(children)):
                child = children[c]
                addToSpecies(child)

        pool["generation"] = pool["generation"] + 1
        #writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))

def initializePool():
        global pool
        pool = newPool()
        for j in range(Population):
                basic = basicGenome()
                addToSpecies(basic)
        initializeRun()


def clearJoypad():
    pass
        # controller = {}
        # for b in range(1,len(ButtonNames)):
        #         controller["P1 " +ButtonNames[b]] = False
        # #joypad.set(controller)

def initializeRun():
        species = pool["species"][pool["currentSpecies"]]
        genome = species["genomes"][pool["currentGenome"]]
        generateNetwork(genome)
        # evaluateCurrent(getInputs())


def evaluateCurrent(inputs):
        species = pool["species"][pool["currentSpecies"]]
        genome = species["genomes"][pool["currentGenome"]]

        return evaluateNetwork(genome["network"], inputs)


def nextGenome():
        pool["currentGenome"] = pool["currentGenome"] + 1
        if pool["currentGenome"] >= len(pool["species"][pool["currentSpecies"]]["genomes"]):
                pool["currentGenome"] = 0
                pool["currentSpecies"] = pool["currentSpecies"]+1
                if pool["currentSpecies"] >= len(pool["species"]):
                        newGeneration()
                        pool["currentSpecies"] = 0


def fitnessAlreadyMeasured():
        species = pool["species"][pool["currentSpecies"]]
        genome = species["genomes"][pool["currentGenome"]]
        return genome["fitness"] != 0




global pool
pool = None
if pool == None :
        initializePool()

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK, state
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        state = "welcome"
        movementInfo = showWelcomeAnimation()
        state = "ingame"
        crashInfo = mainGame(movementInfo)
        state = "gameend"
        showGameOverScreen(crashInfo)

class EventMock():
    def __init__(self):
        self.key = K_SPACE
        self.type = KEYDOWN

def getInput():
    return EventMock()

def getNetworkDecision(input):
    return evaluateCurrent(input)[0]


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        event = getInput()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            # make first flap sound and return values for mainGame
            SOUNDS['wing'].play()
            return {
                'playery': playery + playerShmVals['val'],
                'basex': basex,
                'playerIndexGen': playerIndexGen,
            }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    global birdx,birdy
    species = pool["species"][pool["currentSpecies"]]
    genome = species["genomes"][pool["currentGenome"]]
    generateNetwork(genome)
    fit = 0

    while True:
        fit+=1
        birdx,birdy=playerx,playery
        event = getInput()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        # print playery/10-lowerPipes[0]['y']/10, playerx/10-lowerPipes[0]['x']/10
        pi=0
        for pipe in lowerPipes:
            if pipe['x'] + IMAGES['player'][0].get_width() + 10 > playerx :
                pi = pipe
                break
        if getNetworkDecision([playery/10-pi['y']/10+IMAGES['player'][0].get_height()/10 + 4]) == 1:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                SOUNDS['wing'].play()

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            genome["fitness"] = fit
            if fit > pool["maxFitness"] :
                pool["maxFitness"] = fit
                print "Max Fitness: " + str(math.floor(pool["maxFitness"]))
            print "best fittness "+str(pool["maxFitness"]), "current gen "+str(pool["generation"])
            nextGenome()
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))
        global FPS
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        for inp in pygame.event.get():
            if inp.type ==KEYDOWN and inp.key == K_UP:
                FPS +=20
                print FPS
            elif inp.type ==KEYDOWN and inp.key == K_DOWN:
                FPS -=20
                print FPS



def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        event = getInput()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            if playery + playerHeight >= BASEY - 1:
                return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)
        SCREEN.blit(IMAGES['player'][1], (playerx,playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
