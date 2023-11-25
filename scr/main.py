import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Класс для представления печи
class Oven:
    def __init__(self, start_temp, working_temps, operations):
        self.start_temp = start_temp
        self.working_temps = working_temps
        self.operations = operations

# Класс для представления серии
class Series:
    def __init__(self, temperature, operations):
        self.temperature = temperature
        self.operations = operations

# RL-модель на основе Q-обучения
class QLearningModel:
    def __init__(self, num_ovens):
        # Инициализация Q-таблицы для каждой печи
        self.q_tables = [np.zeros((3,)) for _ in range(num_ovens)]  # 3 действия: -1, 0, 1

    def get_action(self, state):
        # Используем стратегию epsilon-жадности для выбора действия
        epsilon = 0.1
        if np.random.rand() < epsilon:
            return np.random.choice(3) - 1  # Случайное действие: -1, 0, 1
        else:
            return np.argmax(self.q_tables[state])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        # Приводим значения к допустимым пределам
        action = np.clip(action, -1, 1)
        next_state = np.clip(next_state, 0, len(self.q_tables) - 1)

        # Обновление Q-значения по формуле обновления Q-обучения
        self.q_tables[state][action + 1] += alpha * (reward + gamma * np.max(self.q_tables[next_state]) - self.q_tables[state][action + 1])

# Функция для чтения данных из JSON-файла
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Функция для оценки приспособленности с учетом RL-модели
def evaluate_fitness_rl(individual, ovens, series, rl_model):
    series_distribution = {}  # Сохраняем, какая серия идет на какую печь
    for i, oven_index in enumerate(individual):
        series_distribution.setdefault(oven_index, []).append(series[i])

    fitness_score = 0
    for oven_index, series_list in series_distribution.items():
        current_temp = ovens[oven_index % len(ovens)].start_temp

        for series_item in series_list:
            for operation in series_item.operations:
                # Рассчитываем изменение температуры в зависимости от операции
                if operation == "otzhig":
                    current_temp += 30  # Пример: увеличение температуры на 30
                elif operation == "kovka":
                    current_temp -= 20  # Пример: уменьшение температуры на 20
                elif operation == "prokat":
                    current_temp += 15  # Пример: увеличение температуры на 15

            # Рассчитываем оценку приспособленности на основе температуры
            fitness_score += abs(current_temp - series_item.temperature)

            # Обновляем RL-модель
            state = oven_index % len(ovens)
            action = rl_model.get_action(state)
            reward = -abs(current_temp - series_item.temperature)
            next_state = (oven_index + 1) % len(ovens)
            alpha = 0.1  # коэффициент обучения
            gamma = 0.9  # коэффициент дисконтирования
            rl_model.update_q_table(state, action, reward, next_state, alpha, gamma)

    return fitness_score,

# Функция для оценки температур с учетом RL-модели
def evaluate_temperature_rl(individual, ovens, series, rl_model):
    series_distribution = {}  # Сохраняем, какая серия идет на какую печь
    for i, oven_index in enumerate(individual):
        series_distribution.setdefault(oven_index, []).append(series[i])

    temperatures = np.zeros((len(series), len(ovens)))

    for oven_index, series_list in series_distribution.items():
        current_temp = ovens[oven_index % len(ovens)].start_temp

        for i, series_item in enumerate(series_list):
            for operation in series_item.operations:
                # Рассчитываем изменение температуры в зависимости от операции
                if operation == "otzhig":
                    current_temp += 30  # Пример: увеличение температуры на 30
                elif operation == "kovka":
                    current_temp -= 20  # Пример: уменьшение температуры на 20
                elif operation == "prokat":
                    current_temp += 15  # Пример: увеличение температуры на 15

                # Сохраняем температуру
                temperatures[i, oven_index % len(ovens)] = current_temp

                # Обновляем RL-модель
                state = oven_index % len(ovens)
                action = rl_model.get_action(state)
                reward = -abs(current_temp - series_item.temperature)
                next_state = (oven_index + 1) % len(ovens)
                alpha = 0.1  # коэффициент обучения
                gamma = 0.9  # коэффициент дисконтирования
                rl_model.update_q_table(state, action, reward, next_state, alpha, gamma)

    return temperatures

# Функция для инициализации начальной популяции
def generate_initial_population(population_size, num_series, num_ovens):
    return np.random.randint(0, num_ovens, size=(population_size, num_series))

# Функция для выбора лучших родителей на основе турнирного отбора
def select_parents(population, fitness_scores, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(len(population), size=5, replace=False)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        selected_index = tournament_indices[np.argmin(tournament_scores)]
        selected_parents.append(population[selected_index])
    return np.array(selected_parents)

# Функция для кроссовера (одноточечного в данном случае)
def crossover(parents, num_offspring):
    offspring = np.empty((num_offspring, parents.shape[1]), dtype=int)
    crossover_point = parents.shape[1] // 2
    for i in range(0, num_offspring, 2):
        parent1_index, parent2_index = i % len(parents), (i + 1) % len(parents)
        offspring[i, :crossover_point] = parents[parent1_index, :crossover_point]
        offspring[i, crossover_point:] = parents[parent2_index, crossover_point:]
        offspring[i + 1, :crossover_point] = parents[parent2_index, :crossover_point]
        offspring[i + 1, crossover_point:] = parents[parent1_index, crossover_point:]
    return offspring

# Функция для мутации
def mutate(offspring):
    mutation_rate = 0.1
    for child in offspring:
        for i in range(len(child)):
            if np.random.rand() < mutation_rate:
                child[i] = np.random.randint(0, len(child))
    return offspring


# Функция для визуализации результатов
def plot_results(all_fitness, all_temperatures, ovens):
    # График эффективности
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    fitness_values = [fitness[0] for fitness in all_fitness]
    generations = list(range(1, len(fitness_values) + 1))
    plt.plot(generations, fitness_values, label='Лучшая эффективность с RL')
    plt.xlabel('Поколение')
    plt.ylabel('Значение эффективности')
    plt.legend()
    plt.title('Прогресс генетического алгоритма с RL')

    # Тепловая карта загрузки печей
    if all_temperatures and all_temperatures[0].size != 0:
        plt.subplot(1, 3, 2)
        temperatures_combined = np.concatenate(all_temperatures, axis=0)
        plt.imshow(temperatures_combined, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Температура')
        plt.xlabel('Печь')
        plt.ylabel('Серия')
        plt.title('Тепловая карта загрузки печей с RL')

    # График моделирования процесса
    plt.subplot(1, 3, 3)
    for i, temp_array in enumerate(all_temperatures):
        plt.plot(temp_array[:, i % len(ovens)], label=f'Печь {i % len(ovens) + 1}')

    plt.xlabel('Поколение')
    plt.ylabel('Температура')
    plt.legend()
    plt.title('Моделирование процесса в каждой печи с RL')

    plt.tight_layout()
    plt.show()


# Генетический алгоритм с RL-моделью и визуализацией результатов
def genetic_algorithm_with_visualization(num_generations, population_size, num_parents, ovens, series, rl_model):
    num_series = len(series)
    num_ovens = len(ovens)

    # Инициализация начальной популяции
    population = generate_initial_population(population_size, num_series, num_ovens)

    all_fitness = []
    all_temperatures = []

    for generation in range(num_generations):
        # Оценка приспособленности
        fitness_scores = [evaluate_fitness_rl(individual, ovens, series, rl_model) for individual in population]

        # Оценка температур
        temperatures = [evaluate_temperature_rl(individual, ovens, series, rl_model) for individual in population]
        all_temperatures.append(temperatures[-1])

        # Выбор лучших родителей
        parents = select_parents(population, fitness_scores, num_parents)

        # Генерация потомков
        offspring = crossover(parents, population_size - num_parents)

        # Мутация потомков
        offspring = mutate(offspring)

        # Объединение родителей и потомков
        population = np.concatenate((parents, offspring), axis=0)

        # Сохранение лучшего решения в текущем поколении
        best_solution_index = np.argmin(fitness_scores)
        best_solution = population[best_solution_index]
        all_fitness.append(evaluate_fitness_rl(best_solution, ovens, series, rl_model))

    # Визуализация результатов
    plot_results(all_fitness, all_temperatures, ovens)

# Пример использования с интеграцией RL и визуализацией
folder_path = r'C:/Users/RobotComp.ru/Desktop/hacaton'

# Инициализируем переменную ovens перед циклом
ovens = []

for day_number in range(100):
    file_name = f"day-{day_number}.json"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        data = read_json_file(file_path)
        ovens = [Oven(**oven_data) for oven_data in data['ovens']]
        series = [Series(**series_data) for series_data in data['series']]

        # Создаем RL-модель
        rl_model = QLearningModel(num_ovens=len(ovens))

        # Запускаем генетический алгоритм с RL и визуализацией
        genetic_algorithm_with_visualization(num_generations=50, population_size=50, num_parents=20, ovens=ovens, series=series, rl_model=rl_model)

    else:
        print(f"Файл {file_name} не найден.")


