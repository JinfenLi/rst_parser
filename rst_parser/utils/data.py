class_weights = [0.00938, 0.03098, 0.03709, 0.06717, 0.08764, 0.01896, 0.0036, 0.04902, 0.04874, 0.02901, 0.00837, 0.12859, 0.11323, 0.13398, 0.03791, 0.09478, 0.09139, 0.01015]

dataset_info = {
    'rst_dt': {
        'act_classes': ["Shift", "Reduce"],
        'nuc_classes': ["NS", "NN", "SN"],
        'num_act_classes': 2,
        'num_nuc_classes': 3,
        'num_rel_classes': 18,
        'rel_classes': ["Attribution", "Background", "Cause", "Comparison", "Condition", "Contrast", "Elaboration", "Enablement", "Evaluation",
                    "Explanation", "Joint", "Manner-Means", "Topic-Comment", "Summary", "Temporal", "Topic-Change", "Textual-Organization", "Same-Unit"],
        'max_length': {
            'bert-base-uncased': 20,
        },
        'num_special_tokens': 2,
    }
}


monitor_dict = {
    'rst_dt': 'dev_span_f1_metric_epoch'
}

data_keys = ['item_idx', 'edu_input_ids', 'edu_attention_masks', 'spans', 'actions', 'forms', 'relations', 'glove_embs', 'character_ids']

action_dict = {0: 'Shift', 1: 'Reduce'}

nuc_dict = {0: 'SN', 1: 'NN', 2: 'NS'}

main_subclass_dict = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative'],
    'Background': ['background', 'background-e', 'circumstance', 'circumstance-e'],
    'Cause': ['cause', 'cause-result', 'result', 'result-e', 'consequence', 'consequence-n-e', 'consequence-n',
              'consequence-s-e', 'consequence-s'],
    'Comparison': ['comparison', 'comparison-e', 'preference', 'preference-e', 'analogy', 'analogy-e', 'proportion'],
    'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise'],
    'Contrast': ['contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e'],
    'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific-e',
                    'elaboration-general-specific', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e'],
    'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'interpretation-n',
                   'interpretation-s-e', 'interpretation-s', 'interpretation', 'conclusion', 'comment', 'comment-e',
                   'comment-topic'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'reason',
                    'reason-e'],
    'Joint': ['list', 'disjunction'],
    'Manner-Means': ['manner', 'manner-e', 'means', 'means-e'],
    'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'question-answer',
                      'question-answer-n', 'question-answer-s', 'statement-response', 'statement-response-n',
                      'statement-response-s', 'topic-comment', 'comment-topic', 'rhetorical-question'],
    'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e'],
    'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'temporal-same-time',
                 'temporal-same-time-e', 'sequence', 'inverted-sequence'],
    'Topic-Change': ['topic-shift', 'topic-drift'],
    'Textual-Organization': ['textualorganization'],
    'Same-Unit': ['same-unit'],
    # 'span': ['span']
}


sub_mainclass_dict = {}
for i, (main_class, sub_classes) in enumerate(main_subclass_dict.items()):
    for sub_class in sub_classes:
        sub_mainclass_dict[sub_class] = (main_class, i)

