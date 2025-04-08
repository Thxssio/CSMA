import random
import math
import collections
import matplotlib.pyplot as plt

# Ajuste de tempo de simulação para evitar travamento
maxSimulationTime = 300

class Node:
    def __init__(self, location, A):
        self.queue = collections.deque(self.generate_queue(A))
        self.location = location 
        self.collisions = 0
        self.wait_collisions = 0
        self.MAX_COLLISIONS = 10

    def collision_occured(self, R):
        self.collisions += 1
        if self.collisions > self.MAX_COLLISIONS:
            return self.pop_packet()

        backoff_time = self.queue[0] + self.exponential_backoff_time(R, self.collisions)
        for i in range(len(self.queue)):
            if backoff_time >= self.queue[i]:
                self.queue[i] = backoff_time
            else:
                break

    def successful_transmission(self):
        self.collisions = 0
        self.wait_collisions = 0

    def generate_queue(self, A):
        packets = []
        arrival_time_sum = 0
        while arrival_time_sum <= maxSimulationTime:
            arrival_time_sum += get_exponential_random_variable(A)
            packets.append(arrival_time_sum)
        return sorted(packets)

    def exponential_backoff_time(self, R, general_collisions):
        rand_num = random.random() * (pow(2, general_collisions) - 1)
        return rand_num * 512/float(R)  

    def pop_packet(self):
        if self.queue:
            self.queue.popleft()
        self.collisions = 0
        self.wait_collisions = 0

    def non_persistent_bus_busy(self, R):
        self.wait_collisions += 1
        if self.wait_collisions > self.MAX_COLLISIONS:
            return self.pop_packet()

        backoff_time = self.queue[0] + self.exponential_backoff_time(R, self.wait_collisions)
        for i in range(len(self.queue)):
            if backoff_time >= self.queue[i]:
                self.queue[i] = backoff_time
            else:
                break


def get_exponential_random_variable(param):
    uniform_random_value = 1 - random.uniform(0, 1)
    return (-math.log(1 - uniform_random_value) / float(param))

def build_nodes(N, A, D):
    return [Node(i * D, A) for i in range(N)]

def csma_cd(N, A, R, L, D, S, is_persistent):
    curr_time = 0
    transmitted_packets = 0
    successfuly_transmitted_packets = 0
    nodes = build_nodes(N, A, D)

    while True:
        min_node = None
        min_time = float("inf")
        for node in nodes:
            if node.queue and node.queue[0] < min_time:
                min_node = node
                min_time = node.queue[0]

        if min_node is None:
            break

        curr_time = min_time
        transmitted_packets += 1
        collsion_occurred_once = False

        for node in nodes:
            if node.location != min_node.location and node.queue:
                delta_location = abs(min_node.location - node.location)
                t_prop = delta_location / float(S)
                t_trans = L / float(R)
                will_collide = node.queue[0] <= (curr_time + t_prop)

                if (curr_time + t_prop) < node.queue[0] < (curr_time + t_prop + t_trans):
                    if is_persistent:
                        for i in range(len(node.queue)):
                            if (curr_time + t_prop) < node.queue[i] < (curr_time + t_prop + t_trans):
                                node.queue[i] = (curr_time + t_prop + t_trans)
                            else:
                                break
                    else:
                        node.non_persistent_bus_busy(R)

                if will_collide:
                    collsion_occurred_once = True
                    transmitted_packets += 1
                    node.collision_occured(R)

        if not collsion_occurred_once:
            successfuly_transmitted_packets += 1
            min_node.pop_packet()
        else:
            min_node.collision_occured(R)

        # Limite de pacotes simulados para evitar travamento
        if transmitted_packets > 2000:
            break

    efficiency = successfuly_transmitted_packets / float(transmitted_packets) if transmitted_packets else 0
    throughput = (L * successfuly_transmitted_packets) / float(curr_time + (L / R)) * 1e-6
    return efficiency, throughput


# Parâmetros fixos
D = 10
C = 3 * pow(10, 8)
S = (2 / float(3)) * C
R = 1 * pow(10, 6)
L = 1500
A = 10  # taxa de chegada fixa para esta simulação

# Executar simulações
node_counts = list(range(20, 101, 20))
results_persistent = []
results_non_persistent = []

for N in node_counts:
    eff_p, thr_p = csma_cd(N, A, R, L, D, S, True)
    results_persistent.append((eff_p, thr_p))
    eff_np, thr_np = csma_cd(N, A, R, L, D, S, False)
    results_non_persistent.append((eff_np, thr_np))

# Separar para plotagem
eff_p_vals, thr_p_vals = zip(*results_persistent)
eff_np_vals, thr_np_vals = zip(*results_non_persistent)

# Gráficos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(node_counts, eff_p_vals, marker='o', label="Persistente")
plt.plot(node_counts, eff_np_vals, marker='s', label="Não-persistente")
plt.title("Eficiência vs Número de Nós")
plt.xlabel("Número de Nós")
plt.ylabel("Eficiência")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(node_counts, thr_p_vals, marker='o', label="Persistente")
plt.plot(node_counts, thr_np_vals, marker='s', label="Não-persistente")
plt.title("Throughput vs Número de Nós")
plt.xlabel("Número de Nós")
plt.ylabel("Throughput (Mbps)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
