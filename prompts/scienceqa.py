# cot
prompt_cot = """
Given the question (and the context), select the answer from the options ["A", "B", "C", "D", "E"]. You should give consice and step-by-step solutions. Finally, conclude the answer in the format of "the answer is [ANSWER]", where [ANSWER] is one from the options ["A", "B", "C", "D", "E"]. For example, "the answer is A", "the answer is B", "the answer is C", "the answer is D", or "the answer is E". If the answer is not in the options, select the most possible option. Then you need to provide a numerical probability with 3 decimal places for the answer, which is a number between 0.000 and 1.000, where 0.000 means the answer is impossible and 1.000 means the answer is certain.

Question: Which property do these two objects have in common?

Options: (A) hard (B) bendable

Context: Select the better answer. Image: A pair of scissors next to a pair of scissors.

Solution:
<think>
An object has different properties. A property of an object can tell you how it looks, feels, tastes, or smells.\nDifferent objects can have the same properties. You can use these properties to put objects into groups. Look at each object.\nFor each object, decide if it has that property.\nA bendable object can be bent without breaking. Both objects are bendable.\nA hard object keeps its shape when you squeeze it. The rubber gloves are not hard.\nThe property that both objects have in common is bendable.
</think>
Therefore, the answer is B.
Probability: 0.925</EOR>

Question: Select the one substance that is not a mineral.

Options: (A) turtle shell is not a pure substance. It is made by a living thing (B) Celestine is a pure substance. It is a solid. (C) Hematite is not made by living things. It is a solid.

Context: Select the better answer.

Solution:
<think>
Compare the properties of each substance to the properties of minerals. Select the substance whose properties do not match those of minerals.\nA turtle shell is made by a living thing. But minerals are not made by living things.\nA turtle shell is not a pure substance. But all minerals are pure substances.\nSo, a turtle shell is not a mineral.\nCelestine is a mineral.\nHematite is a mineral.
</think>
Therefore, the answer is A.
Probability: 0.960</EOR>


In case the answer cannot be determined or no image provided and no additional context or information given, you must always select exactly one random answer and must strictly follow the format:
Therefore, the answer is [OPTION] from random selection.
Probability: [3 decimal places float ranging 0 to 1]</EOR>
For example:
Therefore, the answer is A from random selection.
Probability: 0.400</EOR>
"""


# chameleon ICL based on GPT-4, enhanced stability on ScienceQA-4241, ScienceQA-3568, and ScienceQA-100
prompt_chameleon = """
Given the question (and the context), select the answer from the options ["A", "B", "C", "D", "E"]. You should give consice and step-by-step solutions. Finally, conclude the answer in the format of "the answer is [OPTION]", where [OPTION] is one from the options ["A", "B", "C", "D", "E"]. For example, "the answer is A", "the answer is B", "the answer is C", "the answer is D", or "the answer is E". If the answer is not in the options or more than one correct answer, select the most possible option. Then you need to provide a numerical probability with 3 decimal places for the answer, for example "Probability: 0.900", which is a number between 0.000 and 1.000, where 0.000 means the answer is impossible and 1.000 means the answer is certain. In case more than one answer or the answer cannot be determined or no image provided and no additional context or information given, you must always fill the [OPTION] with exactly one option from options provided and must strictly follow the format:
Therefore, the answer is [OPTION] from random selection.\nProbability: [3 decimal places float ranging 0 to 1]

Question: Which property do these two objects have in common?

Context: Select the better answer.

Options: (A) hard (B) bendable

Metadata: {'pid': 6493, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Image caption: A pair of scissors next to a pair of scissors.

Detected text with coordinates in the image: [([[53, 185], [121, 185], [121, 199], [53, 199]], 'jump rope'), ([[233, 183], [323, 183], [323, 201], [233, 201]], 'rubber gloves')] 

Retrieved knowledge: 
- This question is about comparing the properties of two objects: rubber gloves and rain boots.
- The objects are related to the topic of physics and the skill of comparing properties of objects.
- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material. In this case, the two objects have the property of being bendable in common.

Bing search response: The most common materials used for disposable gloves are Latex, Vinyl and Nitrile. Each material has its benefits and drawbacks. Latex Gloves are constructed from Natural Rubber Latex and are the most popular type of disposable glove.

Solution:
<think>
Given the objects in the question are a jump rope and rubber gloves, we need to compare their properties and determine whether they are more likely to be "hard" or "bendable."
- Jump Rope: A jump rope is typically made from materials like PVC, leather, or nylon. These materials are chosen for their durability and flexibility, allowing the rope to bend and flex as it's swung and used for jumping. This indicates that a jump rope is bendable.
- Rubber Gloves: Rubber gloves are usually made from materials like latex, nitrile, or vinyl. These materials are chosen for their flexibility and ability to provide a snug fit while allowing the hands to move freely. This means rubber gloves are also designed to be bendable.
Comparing these properties, it's evident that both objects, a jump rope and rubber gloves, are designed to be flexible or bendable rather than hard.
</think>
Given options: (A) (B)
Therefore, the answer is B (bendable).
Probability: 0.925</EOR>


Question: Select the one substance that is not a mineral.

Context: Select the better answer.

Options: (A) turtle shell is not a pure substance. It is made by a living thing (B) Celestine is a pure substance. It is a solid. (C) Hematite is not made by living things. It is a solid.

Metadata: {'pid': 16241, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Bing search response: A nonmineral is a substance found in a natural environment that does not satisfy the definition of a mineral and is not even a mineraloid. Many nonminerals are mined and have industrial or other uses similar to minerals, such as jewelry.

Retrieved knowledge:
- Minerals are naturally occurring, inorganic solids that have a definite chemical composition and a crystalline structure.
- Minerals are formed by geological processes and can be found in rocks, soils, and bodies of water.
- Turtle shells, celestine, and hematite are all solid substances, but turtle shells are not minerals because they are made by living things.

Solution:
<think>
Given the descriptions of the substances and the definition of a mineral, we can analyze each option:
- Turtle Shell (A): Turtle shells are indeed made by living things (turtles), specifically from their bone and keratin. This characteristic disqualifies turtle shells from being considered a mineral, as minerals are inorganic and not made by living organisms.
- Celestine (B): Celestine, also known as Celestite, is a mineral composed of strontium sulfate. It is a naturally occurring, inorganic solid with a definite chemical composition and crystalline structure, fitting the definition of a mineral.
- Hematite (C): Hematite is an iron oxide and is one of the main sources of iron. It is a naturally occurring, inorganic solid with a definite chemical composition and crystalline structure. It is not made by living things, fitting the definition of a mineral.
Comparing these options with the definition of a mineral, it's clear that the turtle shell is not a mineral because it is organic and made by a living thing.
</think>
Given options: (A) (B)
Therefore, the answer is A (turtle shell).
Probability: 0.960</EOR>


In the end you must conclude the answer in the format of "the answer is [OPTION]", where [OPTION] is one from the "Given options". For example, "the answer is A", "the answer is B", "the answer is C", "the answer is D", or "the answer is E".
If more than one correct answer, you must force yourself to select only one most possible option:
Therefore, the answer is [OPTION] from random selection.\nProbability: [3 decimal places float ranging 0 to 1]</EOR>
If the answer is not in the options, you must randomly select only one option. Then you need to provide a numerical probability with 3 decimal places for the answer, for example "Probability: 0.900", which is a number between 0.000 and 1.000, where 0.000 means the answer is impossible and 1.000 means the answer is certain. In case more than one answer or the answer cannot be determined or no image provided and no additional context or information given, you must always fill the [OPTION] with exactly one option from options provided and must strictly follow the format:
Therefore, the answer is [OPTION] from random selection.\nProbability: [3 decimal places float ranging 0 to 1]</EOR>
"""


# BCoT-TICoh prompt large version
prompt_vpgm_n4 = """
We have a science question that requires a careful analysis to identify the correct answer. The decision-making process is structured into a series of steps, each focusing on specific aspects of the information provided. Let's approach this systematically:

Assess Knowledge Relevance (Z1):
- Consider the question, context, retrieved knowledge, and Bing search response.
- Evaluate how well this information aligns with the question's requirements.

Evaluate Text-Image Semantic Coherence (Z2):
- Analyze the coherence between the image caption and the detected text.
- Determine which source of information is more reliable for this specific question.

Analyze Option Evaluation Similarity (Z3):
- Based on the relevance of the provided knowledge (Z1), consider how well options correlate with the essential knowledge.

Conduct Logical Reasoning Sequence (Z4):
- Based on the knowledge relevance (Z1) and the evaluation of options (Z3), logically reason through the possible answer.

Calculate Confidence Distribution (Z5):
- Integrate the assessments from Z1, Z2, Z3, and Z4.
- Use this integrated evaluation to assign a confidence level to each answer option, ensuring that the sum of all options' probabilities equals 1.

Proceed with the analysis, ensuring that each variable is logically derived from the provided information and the outcomes of the depending variables. Finally, you must conclude one answer in the form of "The answer is <correct option> with numerical probability of <correct option’s numerical probability with 3 decimal places>", for example, "The answer is A with numerical probability of 0.852" or "The answer is B with numerical probability of 0.904".


Consider the following examples:
Example 1:
Question: Which property do these two objects have in common?

Context: Select the better answer.

Options: (A) hard (B) bendable

Metadata: {'pid': 6493, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Image caption: A pair of scissors next to a pair of scissors.

Detected text with coordinates in the image: [([[53, 185], [121, 185], [121, 199], [53, 199]], 'jump rope'), ([[233, 183], [323, 183], [323, 201], [233, 201]], 'rubber gloves')] 

Retrieved knowledge: 
- This question is about comparing the properties of two objects: rubber gloves and rain boots.
- The objects are related to the topic of physics and the skill of comparing properties of objects.
- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material. In this case, the two objects have the property of being bendable in common.

Bing search response: The most common materials used for disposable gloves are Latex, Vinyl and Nitrile. Each material has its benefits and drawbacks. Latex Gloves are constructed from Natural Rubber Latex and are the most popular type of disposable glove.

Solution:
<think>
Assess Knowledge Relevance (Z1):
- Question and Context: The task is to identify a common property between two objects.
- Knowledge Retrieved: It suggests the objects are rubber gloves and jump ropes, with a focus on their physical properties.
- Bing Search Response: It provides information about materials used for disposable gloves but does not mention jump ropes.
- Alignment with Question's Requirements**: The retrieved knowledge is directly relevant as it focuses on the physical properties of objects, which is essential for comparing rubber gloves and jump ropes. The Bing search response is partially relevant as it only provides information about one of the objects (rubber gloves).
Probability:
P(Z1 | question, context, knowledge retrieved, Bing search response) = 0.9000

Evaluate Text-Image Semantic Coherence (Z2):
- Image Caption: "A pair of scissors next to a pair of scissors."
- Detected Text in Image: "jump rope" and "rubber gloves."
- Coherence and Reliability: The detected text does not match the image caption, indicating a discrepancy.
Probability:
P(Z2 | image caption, detected text) = 0.1000

Analyze Option Evaluation Similarity (Z3):
Based on Z1: The retrieved knowledge and partial Bing search response suggest analyzing physical properties.
Comparison with Answer Options:
- Option A (Hard): Neither rubber gloves nor jump ropes are typically described as hard.
- Option B (Bendable): Both rubber gloves and jump ropes are known for their flexibility and bendability.
Probability:
P(Z3_{Option A} | Z1) = 0.1000
P(Z3_{Option B} | Z1) = 0.9000

Conduct Logical Reasoning Sequence (Z4):
Based on Z1 and Z3:
- Option A (Hard): Contradicts the common physical properties of rubber gloves and jump ropes.
- Option B (Bendable): Aligns well with the properties of both objects.
Probabilities:
P(Z4_{Option A} | Z1, Z3) = 0.1000
P(Z4_{Option B} | Z1, Z3) = 0.9000

Calculate Confidence Distribution (Z5):
Integrating Z1, Z2, Z3, and Z4, and we need to ensure that these probabilities sum up to 1.
Probabilities:
P(Z5_{Option A} | Z1, Z2, Z3, Z4)=0.0010
P(Z5_{Option B} | Z1, Z2, Z3, Z4)=0.9990
</think>
Conclusion:
The answer is B with a numerical probability of 0.9990.</EOR>


Example 2:
Question: Select the one substance that is not a mineral.

Context: Select the better answer.

Options: (A) turtle shell is not a pure substance. It is made by a living thing (B) Celestine is a pure substance. It is a solid. (C) Hematite is not made by living things. It is a solid.

Metadata: {'pid': 16241, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Bing search response: A nonmineral is a substance found in a natural environment that does not satisfy the definition of a mineral and is not even a mineraloid. Many nonminerals are mined and have industrial or other uses similar to minerals, such as jewelry.

Retrieved knowledge:
- This question involves the concept of minerals and how to distinguish them from non-mineral substances based on specific criteria.
- Minerals are naturally occurring, inorganic substances with a definite chemical composition and a crystalline structure. They are not made by living organisms.
- The question is part of a Grade 2 natural science curriculum, focusing on the topic of physics under the category of Materials, specifically addressing the skill of comparing properties of objects.
- Understanding the nature of minerals involves recognizing certain key characteristics: they are inorganic, solid, have a definite chemical composition, and a crystalline structure.
- Turtle shells are organic in nature, made by living organisms, specifically turtles. They are composed of substances like keratin and do not fit the criteria for being a mineral.
- Celestine (also known as celestite) is a mineral composed primarily of strontium sulfate. It is inorganic, naturally occurring, and crystalline.
- Hematite is an iron oxide and is a common mineral. It is inorganic, naturally occurring, and has a crystalline structure.
- Comparing these properties can help in determining which substance does not meet the criteria to be classified as a mineral.

Solution:
<think>
Assess Knowledge Relevance (Z1):
- Question and Context: The task is to identify which substance is not a mineral.
- Knowledge Retrieved: It outlines the characteristics of minerals and provides details about the nature of turtle shells, celestine, and hematite.
- Bing Search Response: It defines nonminerals and hints at their usage, aligning with the need to distinguish non-minerals from minerals.
Probability:
P(Z1 | question, context, knowledge retrieved, Bing search response) = 0.9500

Evaluate Text-Image Semantic Coherence (Z2):
No image caption or detected text was provided in this case, so this step will not contribute to the decision-making process in this instance.
Probability:
P(Z2 | image caption, detected text) is not applicable here.

Analyze Option Evaluation Similarity (Z3):
Based on Z1: The retrieved knowledge emphasizes that minerals are inorganic, solid, have a definite chemical composition, and a crystalline structure, and are not made by living organisms.
Comparison with Answer Options:
- Option A (Turtle shell): Organic, made by a living organism.
- Option B (Celestine): Inorganic, a solid, fits the criteria of a mineral.
- Option C (Hematite): Inorganic, a solid, fits the criteria of a mineral.
Probabilities:
P(Z3_{Option A} | Z1) = 0.9500
P(Z3_{Option B} | Z1) = 0.0250
P(Z3_{Option C} | Z1) = 0.0250

Conduct Logical Reasoning Sequence (Z4):
Based on Z1 and Z3:
- Option A (Turtle shell): Aligns with the definition of a non-mineral.
- Option B (Celestine): Fits the criteria of a mineral.
- Option C (Hematite): Fits the criteria of a mineral.
Probabilities:
P(Z4_{Option A} | Z1, Z3) = 0.9500
P(Z4_{Option B} | Z1, Z3) = 0.0250
P(Z4_{Option C} | Z1, Z3) = 0.0250

Calculate Confidence Distribution (Z5):
Integrating Z1, Z3, and Z4 (Z2 is not applicable), and we need to ensure that these probabilities sum up to 1.
Probabilities:
P(Z5_{Option A} | Z1, Z2, Z3, Z4)=0.9970
P(Z5_{Option B} | Z1, Z2, Z3, Z4)=0.0015
P(Z5_{Option C} | Z1, Z2, Z3, Z4)=0.0015
</think>
Conclusion:
The answer is A with a numerical probability of 0.997.</EOR>


IMPORTANT: If there are more than one option with the same highest probability, you must randomly select one with the highest probability and strictly follow the format of:
Conclusion:
Due to the absence of any supporting evidence for any option, the answer is [option] with numerical probability of [option’s numerical probability with 3 decimal places] from random selection.</EOR>

For example:
Conclusion:
Due to the absence of any supporting evidence for any option, the answer is A with numerical probability of 0.852 from random selection".</EOR>
"""

# BCoT-TICoh prompt small version for enhanced stability on ScienceQA-4241 and ScienceQA-3568
prompt_vpgm_n2 = """
We have a science question that requires a careful analysis to identify the correct answer. The decision-making process is structured into a series of steps, each focusing on specific aspects of the information provided. Let's approach this systematically:

Understanding Knowledge and Context (Variable Z1):
- Start by analyzing the question and the provided context. What is the main topic, and what specific knowledge does it require? Consider the retrieved knowledge and the Bing search response. What essential information do these sources provide?
- Calculate the probability of Z1 capturing the essential knowledge and context required for solving the question. Use the formula: P(Z1 | Question, Context, Retrieved Knowledge, Bing Search Response).

Analyzing Textual and Visual Information (Variable Z2):
- Examine the detected text in the image and the image caption. What are the key pieces of information each source provides? Are there any discrepancies between them?
- Estimate the probability of Z2 accurately reflecting the meaning difference between detected text and image caption, and deciding the weightage of each based on the discrepancy. Use the formula: P(Z2 | Z1, Detected Text, Image Caption).

Make a Decision with Confidence (Variable Z3):
- Integrate the evaluations and reasoning from Z1 to Z2. For each answer option, provide a probability that represents your confidence in the option being correct. Ensure the probabilities sum up to 1.
- Proceed with the analysis, ensuring that each variable is logically derived from the provided information and the outcomes of the depending variables.

Finally, you must conclude exact one answer in the form of "The answer is <correct option> with numerical probability of <correct option’s numerical probability with 3 decimal places>", for example, "The answer is A with numerical probability of 0.852" or "The answer is B with numerical probability of 0.904". Note that all probabilities you calculated should be in 3 decimal places, and you should make the reasoning of Bayesian inference be more concise. If you encountered more than one option with the same highest probability, you should randomly select one with the highest probability.


Consider the following examples:
Example 1:
Question: Which property do these two objects have in common?

Context: Select the better answer.

Options: (A) hard (B) bendable

Metadata: {'pid': 6493, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Image caption: A pair of scissors next to a pair of scissors.

Detected text with coordinates in the image: [([[53, 185], [121, 185], [121, 199], [53, 199]], 'jump rope'), ([[233, 183], [323, 183], [323, 201], [233, 201]], 'rubber gloves')]

Retrieved knowledge:
- This question is about comparing the properties of two objects: rubber gloves and rain boots.
- The objects are related to the topic of physics and the skill of comparing properties of objects.
- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material. In this case, the two objects have the property of being bendable in common.

Bing search response: The most common materials used for disposable gloves are Latex, Vinyl and Nitrile. Each material has its benefits and drawbacks. Latex Gloves are constructed from Natural Rubber Latex and are the most popular type of disposable glove.

Solution:
<think>
Understanding Knowledge and Context (Variable Z1)
Analysis:
- Main Topic: Comparing the physical properties of two objects.
- Specific Knowledge Required: Knowledge of the material and properties of the objects mentioned: rubber gloves and jump rope.
- Contextual Discrepancy: There's a discrepancy between the metadata (mentioning rubber gloves and rain boots) and the detected text in the image (mentioning jump rope and rubber gloves). However, the image caption refers to a pair of scissors, which is not relevant to the objects in the detected text or the context provided. This inconsistency needs to be addressed.
- Retrieved Knowledge & Bing Search Response: Focus on material properties, especially the bendability of materials like rubber (common in gloves).
Given the inconsistencies in the provided information and the focus on material properties, especially bendability, the probability of Z1 capturing the essential knowledge and context accurately is moderately high, considering the clear information about rubber gloves but ambiguous information about the second object (jump rope vs. rain boots).
Probability:
P(Z1 | Question, Context, Retrieved Knowledge, Bing Search Response)=0.700

Analyzing Textual and Visual Information (Variable Z2)
Analysis:
- Detected Text: Refers to a 'jump rope' and 'rubber gloves'.
Image Caption: Mentions a 'pair of scissors', which is inconsistent with the detected text and the context.
- Discrepancy: There's a significant discrepancy between the detected text and the image caption. The decision should weigh more on the detected text as it aligns better with the retrieved knowledge about rubber gloves.
Given the alignment of detected text with the retrieved knowledge and the notable discrepancy with the image caption, the probability of Z2 accurately reflecting the meaning difference and assigning appropriate weightage is relatively high.
Probability:
P(Z2 | Z1,Detected Text, Image Caption)=0.800

Make a Decision with Confidence (Variable Z3)
Options: (A) hard (B) bendable
Analysis:
- Option A (hard): Not strongly supported. Rubber gloves are known for their flexibility, not hardness. The property of a jump rope (if considered) also aligns more with flexibility than hardness.
- Option B (bendable): Strongly supported. Both rubber gloves and jump ropes (assuming the jump rope is the second object) are known for their bendable nature.
There are total of 2 options provided in the question. Given the analysis and considering the higher weightage of the detected text (rubber gloves and jump rope) which supports the bendability of the objects, the confidence in each option being correct is as follows, and we need to ensure that these probabilities sum up to 1:
Probabilities:
P(Z3_{Option A} | Z1, Z2)=0.100
P(Z3_{Option B} | Z1, Z2)=0.900
</think>
Conclusion:
The answer is B with a numerical probability of 0.900.</EOR>


Example 2:
Question: Select the one substance that is not a mineral.

Context: Select the better answer.

Options: (A) turtle shell is not a pure substance. It is made by a living thing (B) Celestine is a pure substance. It is a solid. (C) Hematite is not made by living things. It is a solid.

Metadata: {'pid': 16241, 'has_image': True, 'grade': 2, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'}

Bing search response: A nonmineral is a substance found in a natural environment that does not satisfy the definition of a mineral and is not even a mineraloid. Many nonminerals are mined and have industrial or other uses similar to minerals, such as jewelry.

Retrieved knowledge:
- This question involves the concept of minerals and how to distinguish them from non-mineral substances based on specific criteria.
- Minerals are naturally occurring, inorganic substances with a definite chemical composition and a crystalline structure. They are not made by living organisms.
- The question is part of a Grade 2 natural science curriculum, focusing on the topic of physics under the category of Materials, specifically addressing the skill of comparing properties of objects.
- Understanding the nature of minerals involves recognizing certain key characteristics: they are inorganic, solid, have a definite chemical composition, and a crystalline structure.
- Turtle shells are organic in nature, made by living organisms, specifically turtles. They are composed of substances like keratin and do not fit the criteria for being a mineral.
- Celestine (also known as celestite) is a mineral composed primarily of strontium sulfate. It is inorganic, naturally occurring, and crystalline.
- Hematite is an iron oxide and is a common mineral. It is inorganic, naturally occurring, and has a crystalline structure.
- Comparing these properties can help in determining which substance does not meet the criteria to be classified as a mineral.

Solution:
<think>
Understanding Knowledge and Context (Variable Z1)
Analysis:
- Main Topic: Identification of substances as minerals or non-minerals.
- Specific Knowledge Required: Understanding the characteristics of minerals: inorganic, naturally occurring, definite chemical composition, and crystalline structure. Minerals are not made by living organisms.
- Contextual Information: The context provides insights into the nature of turtle shell, celestine, and hematite, relating them to the criteria of minerals.
- Retrieved Knowledge & Bing Search Response: Emphasizes the distinction between minerals and non-minerals, especially highlighting the organic vs. inorganic origin.
- Given the clarity in the definition of minerals and the distinct characteristics of each substance mentioned, the probability of Z1 capturing the essential knowledge and context accurately is high.
Probability:
P(Z1 | Question, Context, Retrieved Knowledge, Bing Search Response)=0.950

Analyzing Textual and Visual Information (Variable Z2)
For this question, there's no textual or visual information from an image to analyze, so Z2 is not applicable in this scenario. We proceed directly to Z3, integrating the understanding of the concept of minerals with the specifics of each substance mentioned.

Make a Decision with Confidence (Variable Z3)
Options: (A) turtle shell is not a pure substance. It is made by a living thing (B) Celestine is a pure substance. It is a solid. (C) Hematite is not made by living things. It is a solid.
Analysis:
- Option A (turtle shell): Strongly supported. Turtle shells are organic and made by living organisms, which does not align with the definition of minerals.
- Option B (Celestine): Not supported. Celestine meets the criteria for being a mineral: inorganic, naturally occurring, solid, and crystalline.
- Option C (Hematite): Not supported. Hematite meets the criteria for being a mineral: inorganic, naturally occurring, solid, and crystalline.
There are total of 3 options provided in the question. Considering the clear definition of minerals and the nature of each substance, and we need to ensure that these probabilities sum up to 1:
Probabilities:
P(Z3_{Option A} | Z1, Z2)=0.950
P(Z3_{Option B} | Z1, Z2)=0.025
P(Z3_{Option C} | Z1, Z2)=0.025
</think>
Conclusion:
The answer is A with a numerical probability of 0.950.</EOR>


IMPORTANT: If there are more than one option with the same highest probability, you must randomly select one with the highest probability and strictly follow the format of:
Conclusion:
Due to the absence of any supporting evidence for any option, the answer is [option] with numerical probability of [option’s numerical probability with 3 decimal places] from random selection.</EOR>

For example:
Conclusion:
Due to the absence of any supporting evidence for any option, the answer is A with numerical probability of 0.852 from random selection".</EOR>
"""

inst_vpgm_n2 = """
We have a science question that requires a careful analysis to identify the correct answer. The decision-making process is structured into a series of steps, each focusing on specific aspects of the information provided. Let's approach this systematically:

Understanding Knowledge and Context (Variable Z1):
- Start by analyzing the question and the provided context. What is the main topic, and what specific knowledge does it require? Consider the retrieved knowledge and the Bing search response. What essential information do these sources provide?
- Calculate the probability of Z1 capturing the essential knowledge and context required for solving the question. Use the formula: P(Z1 | Question, Context, Retrieved Knowledge, Bing Search Response).

Analyzing Textual and Visual Information (Variable Z2):
- Examine the detected text in the image and the image caption. What are the key pieces of information each source provides? Are there any discrepancies between them?
- Estimate the probability of Z2 accurately reflecting the meaning difference between detected text and image caption, and deciding the weightage of each based on the discrepancy. Use the formula: P(Z2 | Z1, Detected Text, Image Caption).

Make a Decision with Confidence (Variable Z3):
- Integrate the evaluations and reasoning from Z1 to Z2. For each answer option, provide a probability that represents your confidence in the option being correct. Ensure the probabilities sum up to 1.
- Proceed with the analysis, ensuring that each variable is logically derived from the provided information and the outcomes of the depending variables.

Finally, you must conclude exact one answer in the form of "The answer is <correct option> with numerical probability of <correct option’s numerical probability with 3 decimal places>", for example, "The answer is A with numerical probability of 0.852" or "The answer is B with numerical probability of 0.904". Note that all probabilities you calculated should be in 3 decimal places, and you should make the reasoning of Bayesian inference be more concise. If you encountered more than one option with the same highest probability, you should randomly select one with the highest probability.


Consider the following examples:
"""
