import random
import numpy as np

# Data for rooms, times, and activities
rooms = {
    "Slater 003": 45,
    "Roman 216": 30,
    "Loft 206": 75,
    "Roman 201": 50,
    "Loft 310": 108,
    "Beach 201": 60,
    "Beach 301": 75,
    "Logos 325": 450,
    "Frank 119": 60,
}

times = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]

activities = {
    "SLA100A": {
        "enrollment": 50,
        "preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "others": ["Numen", "Richards"]
    },
    "SLA100B": {
        "enrollment": 50,
        "preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "others": ["Numen", "Richards"]
    },
    "SLA191A": {
        "enrollment": 50,
        "preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "others": ["Numen", "Richards"]
    },
    "SLA191B": {
        "enrollment": 50,
        "preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "others": ["Numen", "Richards"]
    },
    "SLA201": {
        "enrollment": 50,
        "preferred": ["Glen", "Banks", "Zeldin", "Shaw"],
        "others": ["Numen", "Richards", "Singer"]
    },
    "SLA291": {
        "enrollment": 50,
        "preferred": ["Lock", "Banks", "Zeldin", "Singer"],
        "others": ["Numen", "Richards", "Shaw", "Tyler"]
    },
    "SLA303": {
        "enrollment": 60,
        "preferred": ["Glen", "Zeldin", "Banks"],
        "others": ["Numen", "Singer", "Shaw"]
    },
    "SLA304": {
        "enrollment": 25,
        "preferred": ["Glen", "Banks", "Tyler"],
        "others": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]
    },
    "SLA394": {
        "enrollment": 20,
        "preferred": ["Tyler", "Singer"],
        "others": ["Richards", "Zeldin"]
    },
    "SLA449": {
        "enrollment": 60,
        "preferred": ["Tyler", "Singer", "Shaw"],
        "others": ["Zeldin", "Uther"]
    },
    "SLA451": {
        "enrollment": 100,
        "preferred": ["Tyler", "Singer", "Shaw"],
        "others": ["Zeldin", "Uther", "Richards", "Banks"]
    }
}

facilitators = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

# Each individual in the population represents a schedule with random assignments
def create_random_schedule():
    schedule = {}
    for activity in activities.keys():
        room = random.choice(list(rooms.keys()))
        time = random.choice(times)
        facilitator = random.choice(facilitators)
        schedule[activity] = {"room": room, "time": time, "facilitator": facilitator}
    return schedule

def calculate_fitness(schedule):
    fitness = 0
    for activity, assignment in schedule.items():
        room = assignment["room"]
        time = assignment["time"]
        facilitator = assignment["facilitator"]

        # Room size penalties and rewards
        enrollment = activities[activity]["enrollment"]
        capacity = rooms[room]
        if capacity < enrollment:
            fitness -= 0.5
        elif capacity > 3 * enrollment:
            fitness -= 0.2 if capacity <= 6 * enrollment else 0.4
        else:
            fitness += 0.3

        # Facilitator preference rewards
        if facilitator in activities[activity]["preferred"]:
            fitness += 0.5
        elif facilitator in activities[activity]["others"]:
            fitness += 0.2
        else:
            fitness -= 0.1

    return fitness

def initialize_population(size=500):
    return [create_random_schedule() for _ in range(size)]

def softmax(fitness_scores):
    exp_scores = np.exp(fitness_scores - np.max(fitness_scores))
    return exp_scores / exp_scores.sum()

def select_parents(population, fitness_scores):
    probabilities = softmax(fitness_scores)
    parent_indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[parent_indices[0]], population[parent_indices[1]]

def crossover(parent1, parent2):
    child = {}
    for activity in parent1.keys():
        # Randomly choose attributes from each parent for the child
        child[activity] = {
            "room": random.choice([parent1[activity]["room"], parent2[activity]["room"]]),
            "time": random.choice([parent1[activity]["time"], parent2[activity]["time"]]),
            "facilitator": random.choice([parent1[activity]["facilitator"], parent2[activity]["facilitator"]])
        }
    return child

def mutate(schedule, mutation_rate=0.01):
    for activity in schedule.keys():
        if random.random() < mutation_rate:
            # Randomly change one of the attributes
            attr = random.choice(["room", "time", "facilitator"])
            if attr == "room":
                schedule[activity]["room"] = random.choice(list(rooms.keys()))
            elif attr == "time":
                schedule[activity]["time"] = random.choice(times)
            elif attr == "facilitator":
                schedule[activity]["facilitator"] = random.choice(facilitators)

def genetic_algorithm(num_generations=5000, population_size=500, mutation_rate=0.01):
    population = initialize_population(population_size)
    best_fitness = float("-inf")
    best_schedule = None

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(schedule) for schedule in population]

        # Track the best schedule
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_schedule = population[fitness_scores.index(max_fitness)]

        # Stop condition based on improvement
        if generation >= 5000 and abs(max_fitness - best_fitness) < 0.01 * best_fitness:
            break

        # Generate next generation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        mutation_rate /= 2  # Gradually reduce mutation rate

    return best_schedule, best_fitness

if __name__ == "__main__":
    best_schedule, best_fitness = genetic_algorithm()
    print("Best Schedule Fitness:", round(best_fitness, 3))
    for activity, assignment in best_schedule.items():
        print(f"{activity}: Room: {assignment['room']}, Time: {assignment['time']}, Facilitator: {assignment['facilitator']}")
    
    # Save to a file
    with open("best_schedule.txt", "w") as f:
        f.write(f"Best Schedule Fitness: {best_fitness}\n")
        for activity, assignment in best_schedule.items():
            f.write(f"{activity}: Room: {assignment['room']}, Time: {assignment['time']}, Facilitator: {assignment['facilitator']}\n")
