from smdebug.trials import create_trial
from smdebug_rules import *
trial=create_trial("/tmp/tensors")
rule = LossNotDecreasing(trial, tensor_regex="CrossEntropyLoss_output_0", mode="TRAIN")
from smdebug.rules.rule_invoker import invoke_rule
invoke_rule(rule)

