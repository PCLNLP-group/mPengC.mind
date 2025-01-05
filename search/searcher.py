from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
import json

class Config:
    def __init__(self,
                 micro_batch_size,
                 seq_len,
                 hidden_size,
                 head_num,
                 num_layers,
                 micro_size,
                 vocab_size,
                 mp,
                 dp,
                 pp,
                 device_memory=30):
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.num_layers = num_layers
        self.micro_size = micro_size
        self.vocab_size = vocab_size
        self.mp = mp
        self.dp = dp
        self.pp = pp
        self.dtype_size = 2 # FP16
        self.expand_ratio = 4

        # sub attention mask
        self.use_device_memory = device_memory
        self.device_memory = device_memory * 1024 * 1024 * 1024 - self.dtype_size * self.micro_size * self.seq_len * self.seq_len

    def get_next_embedding_memory(self):
        # top query
        weight_params = self.seq_len * self.hidden_size
        weight_params += self.vocab_size * self.hidden_size / self.mp

        weight_memory = weight_params * 4 * 2
        optimizer_memory = weight_params * 4 * 2

        whole_weight_memory = (weight_memory + optimizer_memory) + self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size / self.mp
        return whole_weight_memory


    def get_prev_embedding_memory(self):
        weight_params = self.vocab_size * self.hidden_size / self.mp
        weight_params += self.seq_len * self.hidden_size

        weight_memory = weight_params * 4 * 2
        optimizer_memory = weight_params * 4 * 2
        whole_weight_memory = (weight_memory + optimizer_memory)
        return whole_weight_memory

    def get_prev_embedding_compute_time(self):
        return 12, 16

    def get_next_embedding_compute_time(self):
        return 22.5, 21.87

    def get_attention_memory(self):
        weight_params = self.hidden_size * self.hidden_size * 3 // self.mp
        weight_params += self.hidden_size * self.hidden_size // self.mp
        weight_memory = weight_params * self.dtype_size
        weight_memory += weight_params * self.dtype_size // self.dp
        optimizer_memory = weight_params * 4 * 2 // self.dp
        weight_memory += weight_params * self.dtype_size

        whole_weight_memory = (optimizer_memory + weight_memory)
        return whole_weight_memory

    def get_attention_forward_memory(self):
        forward_memory = 0.0
        # layer norm
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size * 2 / self.mp
        # compute qkv
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size
        # query, key, value
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size * 3 / self.mp
        # softmax
        forward_memory += self.micro_batch_size * self.head_num * self.seq_len * self.seq_len * 4 * 3 / self.mp
        # bmm
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size / self.mp
        return forward_memory

    def get_linear_memory(self):
        weight_params = self.hidden_size * self.hidden_size * self.expand_ratio // self.mp
        # (weight)linear
        weight_params += self.hidden_size * self.hidden_size * self.expand_ratio // self.mp

        weight_memory = weight_params * self.dtype_size
        weight_memory += weight_params * self.dtype_size // self.dp
        optimizer_memory = weight_params * 4 * 2 // self.dp
        weight_memory += weight_params * self.dtype_size

        whole_weight_memory = (weight_memory + optimizer_memory)
        return whole_weight_memory

    def get_attention_compute_time(self):
        # forward
        # recompute: allgather is overlaped

        return (35, 33 + 53)

    def get_linear_compute_time(self):
        return (19.19, 9 + 33)

    def get_linear_operators(self):
        return {
            OperatorType.LINEAR_INPUT: 1,
            #OperatorType.LINEAR_X: 1,
            OperatorType.LINEAR_MAPPING: 1,
            OperatorType.LINEAR_GELU: 1
            }

    def get_attention_operators(self):
        return {
            #OperatorType.ATTENTION_X: 1,
            OperatorType.ATTENTION_QKV: 3,
            OperatorType.ATTENTION_ATTEN: 1,
            OperatorType.ATTENTION_SOFTMAX: 1,
            OperatorType.ATTENTION_BMM: 1
            }

class OperatorType(str, Enum):
    #ATTENTION_X = "ATTENTION_X"
    ATTENTION_QKV = "ATTENTION_QKV"
    ATTENTION_ATTEN = "ATTENTION_ATTEN"
    ATTENTION_SOFTMAX = "ATTENTION_SOFTMAX"
    ATTENTION_BMM = "ATTENTION_BMM"
    LINEAR_INPUT = "LINEAR_INPUT"
    #LINEAR_X = "LINEAR_X"
    LINEAR_MAPPING = "LINEAR_MAPPING"
    LINEAR_GELU = "LINEAR_GELU"

class IntraStageDPQuery:
    def __init__(self, config:Config, cost_map, divisor, ref_compute_time):
        self.config = config
        self.cost_map = cost_map
        self.divisor = divisor
        self.ref_compute_time = ref_compute_time

        self.cached_result = {}
        self.cached_solution = {}

    def get_intra_stage(self, stage_idx, layer_start, layer_end):
        begin_flag = layer_start == 0
        end_flag = layer_end == self.config.num_layers * 2 + 1

        # generate layer graph from [layer_start, layer_end]
        weight_memory = 0.0
        ftime, btime = 0.0, 0.0
        # process first embedding layer
        if layer_start == 0:
            weight_memory += self.config.get_prev_embedding_memory()
            f, b = self.config.get_prev_embedding_compute_time()
            ftime += f
            btime += b
            layer_start += 1

        if layer_end ==  2 * self.config.num_layers + 1:
            weight_memory += self.config.get_next_embedding_memory()
            f, b = self.config.get_next_embedding_compute_time()
            ftime += f
            btime += b
            layer_end -= 1
        # process other layers
        layer_num = layer_end - layer_start + 1
        atten_num = 0
        if layer_start % 2 == 1: # begin with linear layer
            atten_num = (layer_num + 1) // 2
        else:
            atten_num = layer_num // 2

        key = f"{stage_idx}_{layer_num}"
        if layer_start % 2 == 0: # linear
            key += "_linear"
        else:
            key += "_atten"
        if begin_flag:
            key += "_begin"
        if end_flag:
            key += "_end"

        if key in self.cached_result:
            return self.cached_result[key], self.cached_solution[key]

        weight_memory += atten_num * self.config.get_attention_memory()
        f, b = self.config.get_attention_compute_time()
        ftime += atten_num * f
        btime += atten_num * b
        weight_memory += (layer_num - atten_num) * self.config.get_linear_memory()
        f, b = self.config.get_linear_compute_time()
        ftime += (layer_num - atten_num) * f
        btime += (layer_num - atten_num) * b

        # left memory
        left_memory = (self.config.device_memory - weight_memory)
        left_memory -= self.config.get_attention_forward_memory()
        #print(f"left memory: {left_memory / 1024 / 1024 / 1024} GB")
        left_memory = left_memory / self.divisor / (self.config.pp - stage_idx)

        # reduce default recomputation
        left_memory -= atten_num * self.cost_map[OperatorType.LINEAR_INPUT][0]
        if layer_start % 2 == 0: # begin with linear layer
            left_memory -= self.cost_map[OperatorType.LINEAR_INPUT][0]

        # add default recompute
        if layer_end % 2 == 1: # end with attention layer
            btime -= self.cost_map[OperatorType.LINEAR_INPUT][1]

        # beibao wenti
        operator_count = {
            #OperatorType.ATTENTION_X: 0,
            OperatorType.ATTENTION_QKV: 0,
            OperatorType.ATTENTION_ATTEN: 0,
            OperatorType.ATTENTION_SOFTMAX: 0,
            OperatorType.ATTENTION_BMM: 0,
            OperatorType.LINEAR_INPUT: 0,
            #OperatorType.LINEAR_X: 0,
            OperatorType.LINEAR_MAPPING: 0,
            OperatorType.LINEAR_GELU: 0}
        linear_operators = self.config.get_linear_operators()
        attention_operators = self.config.get_attention_operators()
        for op, num in linear_operators.items():
            operator_count[op] += num * (layer_num - atten_num)
        for op, num in attention_operators.items():
            operator_count[op] += num * atten_num
        if layer_start % 2 == 0:
            operator_count[OperatorType.LINEAR_INPUT] -= 1

        # pre-test:
        test_compute_time = ftime + btime
        for op, num in operator_count.items():
            mem, cost = self.cost_map[op]
            test_compute_time -= cost * num
        if test_compute_time > self.ref_compute_time:
            return None, None

        # beibao wenti
        left_memory = int(left_memory)
        if left_memory < 0:
            return None, None
        op_num = len(operator_count)
        l = np.zeros((op_num + 1, left_memory + 1), np.float32)
        trace = np.zeros((op_num + 1, left_memory + 1), np.int32)

        operator_keys = list(operator_count.keys())
        op_idx = 1
        for op in operator_keys:
            num = operator_count[op]
            mem, cost = self.cost_map[op]
            for i in range(0, left_memory + 1):
                l[op_idx][i] = l[op_idx - 1][i]
                for j in range(1, num + 1):
                    if i - mem * j >= 0:
                        if l[op_idx][i] < l[op_idx - 1][i - mem * j] + cost * j:
                            l[op_idx][i] = l[op_idx - 1][i - mem * j] + cost * j
                            trace[op_idx][i] = j
            op_idx += 1

        final_time = ftime + btime - l[op_num][left_memory]
        recompute_op_dict = None
        record_time = (ftime, btime - l[op_num][left_memory])

        if final_time > self.ref_compute_time:
            final_time = None
            record_time = None
        else:
            # trace back
            recompute_op_dict = {}
            for i in range(op_num, 0, -1):
                recompute_op_dict[operator_keys[i - 1]] = trace[i][left_memory].tolist()
                left_memory -= trace[i][left_memory] * self.cost_map[operator_keys[i - 1]][0]

        self.cached_result[key] = record_time
        self.cached_solution[key] = recompute_op_dict

        return record_time, recompute_op_dict

@dataclass
class UniInterStageDPItem:
    """Items used in intra stage DP algorithm.

    Attributes:
        1. main cost: the computation time and comm time
            of activations.
        2. weight cost: the comm time to all gather and
            reduce scatter gradients
        3. cur_state: (device_mesh, l_s, l_e),
            point to intra stage
        4. next_state: (prev_dn,)
            point to prev item
    """
    main_cost: float = 0.0
    next_warmup_time: float = 0.0
    next_steady_time: float = 0.0
    iter_time: float = 0.0


def stage_dp_algorithm(config:Config, intra_stage_dp_query: IntraStageDPQuery, output_filename):
    layer_num = config.num_layers * 2 + 2

    stage_max = [[None] * layer_num for _ in range(config.pp)]
    prev_pointers = [[-1] * layer_num for _ in range(config.pp)]
    stage_idx = config.pp - 1
    for i in range(layer_num - 1, -1, -1):
        result, _ = intra_stage_dp_query.get_intra_stage(stage_idx, i, layer_num - 1)
        if result is None:
            break
        iter_time = result[0] + result[1]
        new_item = UniInterStageDPItem(
            main_cost=iter_time * config.micro_size,
            iter_time=iter_time)
        new_item.next_steady_time = (config.micro_size - 2) * iter_time
        new_item.next_warmup_time = iter_time
        new_item.next_cool_time = iter_time
        if config.micro_size == 1:
            new_item.next_steady_time = iter_time
            new_item.next_warmup_time = 0
            new_item.next_cool_time = 0
        stage_max[stage_idx][i] = new_item
        #print(f"init: {new_item}")

    for s_n in range(config.pp - 2, -1, -1):
        begin = time.time()
        for l_s in range(layer_num - (config.pp - s_n), -1, -1):
            for l_m in range(l_s, layer_num - (config.pp - s_n) + 1):
                #print(f"process stage_idx: {s_n}, l_s: {l_s}, l_m: {l_m}")
                cur_item = stage_max[s_n][l_s]
                #print(f"cur_item: {cur_item}")
                next_item = stage_max[s_n + 1][l_m + 1]
                #print(f"next_item: {next_item}")
                if next_item is None:
                    continue
                mid_item, _ = intra_stage_dp_query.get_intra_stage(s_n, l_s, l_m)
                if mid_item is None:
                    break

                cur_iter_time = mid_item[0] + mid_item[1]

                next_iter_time = next_item.iter_time
                max_iter_time = max(cur_iter_time, next_iter_time)

                iter_num = config.micro_size - (config.pp - s_n)
                if iter_num >= 0:
                    steady_time = max_iter_time * iter_num
                else:
                    steady_time = next_item.next_steady_time

                # warmup stage time
                warmup_time = mid_item[0]
                if iter_num >= 0:
                    warmup_time += max(
                        mid_item[0] * (config.pp - s_n - 1),
                        next_item.next_warmup_time)
                else:
                    warmup_time += max(
                        mid_item[0] * (config.micro_size - 1),
                        next_item.next_warmup_time)

                # cool stage time
                cool_time = mid_item[1]
                if iter_num >= 0:
                    cool_time += max(
                        mid_item[1] * (config.pp - s_n - 1),
                        next_item.next_cool_time)
                else:
                    cool_time += max(
                        mid_item[1] * (config.micro_size - 1),
                        next_item.next_cool_time)

                # new main_cost
                new_main_cost = steady_time + warmup_time + cool_time
                if cur_item is None or cur_item.main_cost > new_main_cost:
                    new_item = UniInterStageDPItem(
                        main_cost=new_main_cost,
                        iter_time = max_iter_time)

                    if iter_num >= 1:
                        new_item.next_steady_time = (steady_time - max_iter_time)
                        if cur_iter_time >= max_iter_time:
                            new_item.next_warmup_time = (
                                warmup_time + mid_item[1])
                            new_item.next_cool_time = (
                                cool_time + mid_item[0])
                        else:
                            new_item.next_warmup_time = (
                                warmup_time + max_iter_time - mid_item[1])
                            new_item.next_cool_time = (
                                cool_time + max_iter_time - mid_item[0])
                    else:
                        new_item.next_steady_time = (
                            steady_time + mid_item[0] + mid_item[1])
                        new_item.next_warmup_time = warmup_time
                        new_item.next_cool_time = cool_time
                    stage_max[s_n][l_s] = new_item
                    prev_pointers[s_n][l_s] = l_m
        end = time.time()
        print(f"stage idx: {s_n}, elapsed time: {end - begin} s")

    # find the best item
    best_solution = stage_max[0][0]
    print(f"best solution: {best_solution}")

    # print result
    solution = []
    stage_idx = 0
    layer_idx = 0
    while stage_idx < config.pp:
        next_layer_idx = prev_pointers[stage_idx][layer_idx]
        if stage_idx == config.pp - 1:
            next_layer_idx = layer_num - 1
        stage_time, stage_solution = intra_stage_dp_query.get_intra_stage(stage_idx, layer_idx, next_layer_idx)
        print(f"stage idx: {stage_idx}, layer: [{layer_idx}, {next_layer_idx}] stage time: {stage_time}")
        stage_solution["stage_idx"] = stage_idx
        stage_solution["layer_range"] = [layer_idx, next_layer_idx]
        solution.append(stage_solution)
        layer_idx = next_layer_idx + 1
        stage_idx += 1

    with open(output_filename, "w") as f:
        json.dump(solution, f, indent=4)


def print_elapsed_time(config:Config,
                       intra_stage_dp_query: IntraStageDPQuery,
                       solution):
    last_solution = solution[config.pp - 1]
    layer_range = last_solution["layer_range"]
    stage_time, _ = intra_stage_dp_query.get_intra_stage(config.pp - 1, layer_range[0], layer_range[1])
    iter_time = stage_time[0] + stage_time[1]
    cur_item = UniInterStageDPItem(
        main_cost=iter_time * config.micro_size,
        iter_time=iter_time)
    cur_item.next_steady_time = (config.micro_size - 2) * iter_time
    cur_item.next_warmup_time = iter_time
    cur_item.next_cool_time = iter_time
    if config.micro_size == 1:
        cur_item.next_steady_time = iter_time
        cur_item.next_warmup_time = 0
        cur_item.next_cool_time = 0

    for stage_idx in range(config.pp - 2, -1, -1):
        stage_config = solution[stage_idx]
        layer_range = stage_config["layer_range"]
        mid_item, _ = intra_stage_dp_query.get_intra_stage(stage_idx, layer_range[0], layer_range[1])

        cur_iter_time = mid_item[0] + mid_item[1]

        next_iter_time = cur_item.iter_time
        max_iter_time = max(cur_iter_time, next_iter_time)

        iter_num = config.micro_size - (config.pp - stage_idx)
        if iter_num >= 0:
            steady_time = max_iter_time * iter_num
        else:
            steady_time = cur_item.next_steady_time

        # warmup stage time
        warmup_time = mid_item[0]
        if iter_num >= 0:
            warmup_time += max(
                mid_item[0] * (config.pp - stage_idx - 1),
                cur_item.next_warmup_time)
        else:
            warmup_time += max(
                mid_item[0] * (config.micro_size - 1),
                cur_item.next_warmup_time)

        # cool stage time
        cool_time = mid_item[1]
        if iter_num >= 0:
            cool_time += max(
                mid_item[1] * (config.pp - stage_idx - 1),
                cur_item.next_cool_time)
        else:
            cool_time += max(
                mid_item[1] * (config.micro_size - 1),
                cur_item.next_cool_time)

        # new main_cost
        new_main_cost = steady_time + warmup_time + cool_time

        new_item = UniInterStageDPItem(
            main_cost=new_main_cost,
            iter_time = max_iter_time)

        if iter_num >= 1:
            new_item.next_steady_time = (steady_time - max_iter_time)
            if cur_iter_time >= max_iter_time:
                new_item.next_warmup_time = (
                    warmup_time + mid_item[1])
                new_item.next_cool_time = (
                    cool_time + mid_item[0])
            else:
                new_item.next_warmup_time = (
                    warmup_time + max_iter_time - mid_item[1])
                new_item.next_cool_time = (
                    cool_time + max_iter_time - mid_item[0])
        else:
            new_item.next_steady_time = (
                steady_time + mid_item[0] + mid_item[1])
            new_item.next_warmup_time = warmup_time
            new_item.next_cool_time = cool_time

        cur_item = new_item
    print(cur_item)


def finegrain_dp_algorithm(config:Config,
                           intra_stage_dp_query: IntraStageDPQuery,
                           output_filename):
    layer_num = config.num_layers * 2 + 2 # add prev embedding and later embedding

    # print result
    solution = []
    stage_idx = config.pp - 1
    layer_idx = layer_num - 1
    while stage_idx >= 0:
        layer_num_per_stage = (config.num_layers // config.pp + 1) * 2
        if stage_idx == config.pp - 1 or stage_idx == 0:
            layer_num_per_stage -= 3
        prev_layer_idx = layer_idx - layer_num_per_stage
        stage_time, stage_solution = intra_stage_dp_query.get_intra_stage(stage_idx, prev_layer_idx + 1, layer_idx)
        print(f"stage idx: {stage_idx}, layer: [{prev_layer_idx + 1}, {layer_idx}] stage time: {stage_time}")
        stage_solution["stage_idx"] = stage_idx
        stage_solution["layer_range"] = [prev_layer_idx + 1, layer_idx]
        solution.append(stage_solution)
        layer_idx = prev_layer_idx
        stage_idx -= 1
    solution.reverse()
    print_elapsed_time(gpt_config, intra_stage_dp_query, solution)

    with open(output_filename, "w") as f:
        json.dump(solution, f, indent=4)

def gcd(a, b):
    while b != 0:
        temp = a % b
        a = b
        b = temp
    return a

gpt_config = Config(
    micro_batch_size=1,
    seq_len=4096,
    hidden_size=12672,
    head_num=96,
    num_layers=20,
    micro_size=64,
    vocab_size=49984,
    mp=4,
    dp=4,
    pp=4,
    device_memory=29)

atten_x = gpt_config.micro_batch_size * gpt_config.seq_len * gpt_config.hidden_size * gpt_config.dtype_size // gpt_config.mp

# Float32
atten_atten = gpt_config.micro_batch_size * gpt_config.head_num * gpt_config.seq_len * gpt_config.seq_len * 2 // gpt_config.mp

# cost of GPT3 model, mp = 4
cost_map_mp4 = {
    #OperatorType.ATTENTION_X: (atten_x * gpt_config.mp, 0.276),
    OperatorType.ATTENTION_QKV: (atten_x, 1.7),
    OperatorType.ATTENTION_ATTEN: (atten_atten, 10.544),
    OperatorType.ATTENTION_SOFTMAX: (atten_atten * 3, 7.5), # mul 3
    OperatorType.ATTENTION_BMM: (atten_x, 1.3),
    OperatorType.LINEAR_INPUT: (atten_x, 9.1),
    #OperatorType.LINEAR_X: (atten_x * gpt_config.mp, 0.276),
    OperatorType.LINEAR_MAPPING: (atten_x * gpt_config.expand_ratio * 2, 5.84), # mul 2
    OperatorType.LINEAR_GELU: (atten_x * gpt_config.expand_ratio, 2.07),
}

max_divisor = gcd(atten_x, atten_atten)
for op, (mem, cost) in cost_map_mp4.items():
    cost_map_mp4[op] = (mem // max_divisor, cost)


# get reference running time
ref_layer_num = gpt_config.num_layers // gpt_config.pp + 1

ref_compute_time = (sum(gpt_config.get_attention_compute_time()) + sum(gpt_config.get_linear_compute_time())) * ref_layer_num
first_stage = ref_compute_time + sum(gpt_config.get_prev_embedding_compute_time())
last_stage = ref_compute_time + sum(gpt_config.get_next_embedding_compute_time())
ref_compute_time = max(first_stage, last_stage)
print(f"ref iteration time: {ref_compute_time}")

intra_stage_dp_query = IntraStageDPQuery(gpt_config, cost_map_mp4, max_divisor, ref_compute_time)

#output_filename = f"BaSys/gpt_{gpt_config.num_layers}layer_{gpt_config.dp}dp_{gpt_config.mp}mp_{gpt_config.pp}pp_{gpt_config.micro_size}micro.json"
#stage_dp_algorithm(gpt_config, intra_stage_dp_query, output_filename)


output_filename = f"BaSys_FineGrain/gpt_{gpt_config.num_layers}layer_{gpt_config.dp}dp_{gpt_config.mp}mp_{gpt_config.pp}pp_{gpt_config.use_device_memory}mem_{gpt_config.micro_size}ms.json"
finegrain_dp_algorithm(gpt_config, intra_stage_dp_query, output_filename)

