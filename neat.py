import math
import random



Inputs = 1
Outputs = 1

Population = 300
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

