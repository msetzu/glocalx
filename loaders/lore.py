import json
from numpy import inf
from models import Rule

def lore_to_glocalx(json_file, info_file):
    """Load file `json_file` and `info_file` and return the loaded JSON
    rules in a list of `Rule` objects preserving the loading order.
    Args:
        json_file (str): Path to the JSON file.
        info_file (str): Path to the info file containing the rules' metadata.
    Returns:
        (list): List of `Rule` objects.
    """
    with open(json_file, 'r') as rules_log, open(info_file, 'r') as info_log:
        rules = json.load(rules_log)
        infos = json.load(info_log)
    class_values = infos['class_values']
    feature_names = infos['feature_names']

    rules = [r for r in rules if len(r) > 0]
    output_rules = []
    for rule in rules:
        consequence = class_values.index(rule['cons'])
        premises = rule['premise']
        features = [feature_names.index(premise['att']) for premise in premises]
        ops = [premise['op'] for premise in premises]
        values = [premise['thr'] for premise in premises]
        values_per_feature = {feature: [val for f, val in zip(features, values) if f == feature]
                              for feature in features}
        ops_per_feature = {feature: [op for f, op in zip(features, ops) if f == feature]
                           for feature in features}

        output_premises = {}
        for f in features:
            values, operators = values_per_feature[f], ops_per_feature[f]
            # 1 value, either <= or >
            if len(values) == 1:
                if operators[0] == '<=':
                    output_premises[f] = (-inf, values[0])
                else:
                    output_premises[f] = (values[0], +inf)
            # 2 values, < x <=
            else:
                output_premises[f] = (min(values), max(values))

        transformed_rule = Rule(premises=output_premises, consequence=consequence, names=feature_names)
        output_rules.append(transformed_rule)

    output_rules = set(output_rules)

    return output_rules
