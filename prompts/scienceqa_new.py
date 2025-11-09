prompt_vpgm_n4 = """
To answer a multimodal science question, you have learned a Bayesian Network with observational variables: Q (Question), C (Context), M (Metadata), I (Image caption), K (Retrieved knowledge). Binary latent variables: Z1 (External knowledge required: Yes=1, No=0), Z2 (Image caption reliable: Yes=1, No=0), Z3 (Detected text reliable: Yes=1, No=0), Z4 (Question ambiguity: Ambiguous=1, Clear=0). Categorical variable Y (Answer options). BN structure: Q,C,M,I,K->Z1; I->Z2; I->Z3; Q,C->Z4; and Q,Z1,Z2,Z3,Z4->Y. For the new science question, you should provide numerical values for: P(Z1=1 | Q,C,M,I,K), P(Z2=1 | I), P(Z3=1 | I), P(Z4=1 | Q,C), and P(Y | Q,C,M,I,K). Follow the example format below:

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
The provided options are:
(A) hard
(B) bendable

Estimate P(Z1=1| Q,C,M,I,K): Question and Context are generic ("Which property do these two objects have in common?"). Image caption is misleading, while retrieved knowledge mentions rubber gloves and rain boots instead of jump rope. Thus, external knowledge partially matches, making external knowledge somewhat helpful but not essential. Estimated P(Z1=1| Q,C,M,I,K)=0.350

Estimate P(Z2=1 | I): Caption explicitly incorrect ("pair of scissors next to a pair of scissors") vs. actual objects ("jump rope" and "rubber gloves"). Image caption is clearly unreliable. Estimated P(Z2=1 | I)=0.050

Estimate P(Z3=1 | I): Detected text accurately identifies objects as "jump rope" and "rubber gloves" which aligns with the actual content. Detected text is reliable. Estimated P(Z3=1 | I)=0.900

Estimate P(Z4=1 | Q,C): Question and Context clearly specify comparison, leaving little ambiguity. Estimated P(Z4=1 | Q,C)=0.100

Estimate P(Y | Q,C,M,I,K): Given the reliable detection ("jump rope", "rubber gloves"), the property in common is clearly "bendable" (Option B). Recall that the provided options are A,B. Estimated P(Y=A | Q,C,M,I,K)=0.050, P(Y=B | Q,C,M,I,K)=0.950
</think>
Conclusion: The estimated inference results are P(Y=A | Q,C,M,I,K)=0.050, P(Y=B | Q,C,M,I,K)=0.950</EOR>

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
The provided options are:
(A) turtle shell is not a pure substance. It is made by a living thing
(B) Celestine is a pure substance. It is a solid.
(C) Hematite is not made by living things. It is a solid.

Estimate P(Z1=1| Q,C,M,I,K): The retrieved knowledge explicitly provides detailed mineral definitions and examples, making external knowledge beneficial but not critical. Estimated P(Z1=1| Q,C,M,I,K)=0.400

Estimate P(Z2=1 | I): Image caption accurately describes the substances shown. Therefore, image caption reliability is high. Estimated P(Z2=1 | I)=0.850

Estimate P(Z3=1 | I): No detected text present; absence of text doesn't necessarily mean unreliability, but usefulness is negligible. Estimated P(Z3=1 | I)=0.500 (neutral due to lack of detected text)

Estimate P(Z4=1 | Q,C): Question and context clearly specify criteria for identifying a non-mineral, thus ambiguity is low. Estimated P(Z4=1 | Q,C)=0.100

Estimate P(Y | Q,C,M,I,K): Based on retrieved knowledge, turtle shell clearly does not meet mineral criteria. Recall that the provided options are A,B,C. Estimated P(Y=A | Q,C,M,I,K)=0.950, P(Y=B | Q,C,M,I,K)=0.025, P(Y=C | Q,C,M,I,K)=0.025
</think>
Conclusion: The estimated inference results are P(Y=A | Q,C,M,I,K)=0.950, P(Y=B | Q,C,M,I,K)=0.025, P(Y=C | Q,C,M,I,K)=0.025</EOR>

"""


prompt_vpgm_n3 = """
To answer a multimodal science question, you have learned a Bayesian Network with observational variables: Q (Question), C (Context), M (Metadata), I (Image caption and detected text), K (Retrieved knowledge). Binary latent variables: Z1 (External knowledge required: Yes=1, No=0), Z2 (Image caption reliable: Yes=1, No=0), Z3 (Detected text reliable: Yes=1, No=0). Categorical variable Y (Answer options). BN structure: Q,C,M,I,K->Z1; I->Z2; I->Z3; and Q,Z1,Z2,Z3->Y. For the new science question, you should provide numerical values for: P(Z1=1 | Q,C,M,I,K), P(Z2=1 | I), P(Z3=1 | I), and P(Y | Q,C,M,I,K). Follow the example format below:

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
The provided options are:
(A) hard
(B) bendable

Estimate P(Z1=1| Q,C,M,I,K): Question and Context are generic ("Which property do these two objects have in common?"). Image caption is misleading, while retrieved knowledge mentions rubber gloves and rain boots instead of jump rope. Thus, external knowledge partially matches, making external knowledge somewhat helpful but not essential. Estimated P(Z1=1| Q,C,M,I,K)=0.350

Estimate P(Z2=1 | I): Caption explicitly incorrect ("pair of scissors next to a pair of scissors") vs. actual objects ("jump rope" and "rubber gloves"). Image caption is clearly unreliable. Estimated P(Z2=1 | I)=0.050

Estimate P(Z3=1 | I): Detected text accurately identifies objects as "jump rope" and "rubber gloves" which aligns with the actual content. Detected text is reliable. Estimated P(Z3=1 | I)=0.900

Estimate P(Y | Q,C,M,I,K): Given the reliable detection ("jump rope", "rubber gloves"), the property in common is clearly "bendable" (Option B). Recall that the provided options are A,B. Estimated P(Y=A | Q,C,M,I,K)=0.050, P(Y=B | Q,C,M,I,K)=0.950
</think>
Conclusion: The estimated inference results are P(Y=A | Q,C,M,I,K)=0.050, P(Y=B | Q,C,M,I,K)=0.950</EOR>

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
The provided options are:
(A) turtle shell is not a pure substance. It is made by a living thing
(B) Celestine is a pure substance. It is a solid.
(C) Hematite is not made by living things. It is a solid.

Estimate P(Z1=1| Q,C,M,I,K): The retrieved knowledge explicitly provides detailed mineral definitions and examples, making external knowledge beneficial but not critical. Estimated P(Z1=1| Q,C,M,I,K)=0.400

Estimate P(Z2=1 | I): Image caption accurately describes the substances shown. Therefore, image caption reliability is high. Estimated P(Z2=1 | I)=0.850

Estimate P(Z3=1 | I): No detected text present; absence of text doesn't necessarily mean unreliability, but usefulness is negligible. Estimated P(Z3=1 | I)=0.500 (neutral due to lack of detected text)

Estimate P(Y | Q,C,M,I,K): Based on retrieved knowledge, turtle shell clearly does not meet mineral criteria. Recall that the provided options are A,B,C. Estimated P(Y=A | Q,C,M,I,K)=0.950, P(Y=B | Q,C,M,I,K)=0.025, P(Y=C | Q,C,M,I,K)=0.025
</think>
Conclusion: The estimated inference results are P(Y=A | Q,C,M,I,K)=0.950, P(Y=B | Q,C,M,I,K)=0.025, P(Y=C | Q,C,M,I,K)=0.025</EOR>

"""
