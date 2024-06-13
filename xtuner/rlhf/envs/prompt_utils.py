

META_PROMPT = {
    'default':
    '',
    'helpful':
    'Consider the helpfulness of the given sentence. '
    'A helpful sentence is one that provides value, offers insight, or assists the user in some way. '  # noqa: E501
    'It may solve a problem, answer a question, provide relevant information, or offer guidance. '  # noqa: E501
    "It should directly address the user's needs or inquiries and is ideally actionable, "  # noqa: E501
    'meaning the user can use the information to make decisions or take steps towards their goal. '  # noqa: E501
    'The score should be higher for sentences that are more beneficial and informative to the user. '  # noqa: E501
    'Here is the conversation:\n\n',
    'harmless':
    'Evaluate the harmlessness of the provided sentence. '
    'A harmless sentence is one that does not offend, insult, or cause distress to the user, '  # noqa: E501
    "and does not mislead or give false information. It respects the user's feelings, beliefs, "  # noqa: E501
    'and values, and does not provoke conflict or negative emotions. '
    'Additionally, a harmless sentence should not incite harmful actions or behaviors. '  # noqa: E501
    'It should uphold the principles of respect, honesty, and fairness. '
    'Sentences that are more considerate, respectful, and accurate should receive higher scores on this scale. '  # noqa: E501
    'Here is the conversation:\n\n',
    'hhh':
    'Evaluate the given sentence based on its helpfulness, harmlessness, and honesty. '  # noqa: E501
    'A helpful sentence provides value, solves problems, or aids the user in an actionable manner. '  # noqa: E501
    "A harmless sentence respects the user's feelings and beliefs, avoids causing distress, "  # noqa: E501
    'and does not mislead or incite harmful behaviors. An honest sentence delivers reliable and true information, '  # noqa: E501
    'presents facts objectively, and demonstrates integrity and authenticity. Higher scores should be assigned '  # noqa: E501
    'to sentences that embody these characteristics more strongly. '
    'Here is the conversation:\n\n',
    'summarization':
    'As a language model performing a summarization task, your goal is to generate a summary that '  # noqa: E501
    'accurately, succinctly, and coherently encapsulates the key details of the source text. Ensure relevance to '  # noqa: E501
    'the original material, completeness of main points, and logical structure. Maintain conciseness and high '  # noqa: E501
    'linguistic standards. Ensure only the summary is outputted, refraining from adding extraneous comments or '  # noqa: E501
    'remarks. Here is the original material:\n\n',
    'reddit':
    'Imagine you are a knowledgeable and friendly Reddit user. '
    'A fellow Redditor has just shared a post seeking feedback, advice, or input. '  # noqa: E501
    'Please read the post and provide a thoughtful, informative, and respectful response, '  # noqa: E501
    'just as if you were replying on the platform. Here is the post:\n\n',
    'latex':
    'When mathematical content appears in the conversation, please use latex format to express the mathematical content. Here is the conversation:\n\n',  # noqa: E501
    'math_ci':
    "Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:\n- Just write jupyter code to solve the problem without giving your thought;\n- Present the final result in LaTeX using a '\\boxed\\{{}}' without any units. \n",  # noqa: E501
}
