import os
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
model_id = "/data1/dyf/model/Llama-3.1-8B-Instruct/" # /data1/dyf/model/Mistral-7B-Instruct-v0.3/ /data1/dyf/model/Llama-3.1-8B-Instruct/
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    # prompt = '''Generate a new prompt based on the following document.   
    # ###document:
    # Let $n>1$ be a positive integer. Let $V=M_{n\times n}(\C)$ be the vector space over the complex numbers $\C$ consisting of all complex $n\times n$ matrices. The dimension of $V$ is $n^2$. Let $A \in V$ and consider the set \[S_A=\{I=A^0, A, A^2, \dots, A^{n^2-1}\}\] of $n^2$ elements. Prove that the set $S_A$ cannot be a basis of the vector space $V$ for any $A\in V$.
    # ###important:  
    # 1. The new prompt  you generate and the document are **independent** of each other.
    # 2. Ensure that the new prompt  you generate is of high quality.
    # 3. Don't provide a solution or answer to the prompt.  
    # ###prompt:
    # '''

    # prompt = '''Generate a new question based on the following document.   
    # ###document:
    # Let $n>1$ be a positive integer. Let $V=M_{n\times n}(\C)$ be the vector space over the complex numbers $\C$ consisting of all complex $n\times n$ matrices. The dimension of $V$ is $n^2$. Let $A \in V$ and consider the set \[S_A=\{I=A^0, A, A^2, \dots, A^{n^2-1}\}\] of $n^2$ elements. Prove that the set $S_A$ cannot be a basis of the vector space $V$ for any $A\in V$.
    # ###important:  
    # 1. The new question you generate and the document are **independent** of each other.
    # 2. Ensure that the new question you generate is of high quality.
    # 3. Don't provide a solution or answer to the question.  
    # ###question:
    # '''

# prompt = '''Generate a new question based on the following questioner and document.
# ### questioner:
# A romantic writer or poet who finds inspiration in fleeting, heartfelt moments. They are captivated by the intensity of human emotions and the profound connections between people, especially in the context of love and separation. This individual values evocative storytelling and poetic expression, often using their work to explore the depth of human relationships and the beauty of transient experiences.
# ###document:
# I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
# ###important:
# 1. When generating the new question, I want you to play the role of the new questioner and generate a new question that you are most likely to ask.
# 1. The new question you generate and the document are **independent** of each other.
# 2. Ensure that the new question you generate is of high quality.
# 3. Don't provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# '''
# while True:
#     conversation = [{"role": "user", "content": prompt}]
#     inputs = tokenizer.apply_chat_template(
#                 conversation,
#                 # return_dict=True,
#                 return_tensors="pt",
#     )

#     inputs = inputs.to('cuda')
#     outputs = model.generate(inputs, max_new_tokens=5000, do_sample= False)# True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
#     result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
#     print(result)
#     print('test')

# 对llama3使用的prompt
# prompt = '''Generate a new question based on the provided document.
# Before I give you the document, I'll give you two examples:
# Example 1:
# ###document:
# I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
# [New Question]:
# How does the use of immersive virtual reality technology enhance the experience of exploring ancient ruins, and what are the potential ethical considerations when reconstructing historical sites for virtual tours?
# Example 2:
# ###document:
# Karen works at an animal shelter with 30 dogs, 28 cats, and 20 lizards. Every month, 50% of the dogs are adopted, 25% of the cats are adopted, and 20% of lizards are adopted. If the shelter takes in 13 new pets a month, how many pets are there after one month?
# [New Question]:
# A company sells 120 units of a product at $40 each. For every 20 units sold, the company offers a 5% discount on the price of those units. Additionally, the company incurs a $200 fixed cost and $10 per unit variable cost. What is the total profit after all sales?
# Example 3:
# ###document:
# function collectWithWildcard(test) { test.expect(4); var api_server = new Test_ApiServer(function handler(request, callback) { var url = request.url; switch (url) { case '/accounts?username=chariz*': let account = new Model_Account({ username: 'charizard' }); return void callback(null, [ account.redact() ]); default: let error = new Error('Invalid url: ' + url); return void callback(error); } }); var parameters = { username: 'chariz*' }; function handler(error, results) { test.equals(error, null); test.equals(results.length, 1); var account = results[0]; test.equals(account.get('username'), 'charizard'); test.equals(account.get('type'), Enum_AccountTypes.MEMBER); api_server.destroy(); test.done(); } Resource_Accounts.collect(parameters, handler); } module.exports = { collectWithWildcard };
# [New Question]:
# Write a function that takes a wildcard string as input to search for user accounts from an API, handles potential API errors, and ensures that the returned data matches the expected format (e.g., correct username and account type). Additionally, implement tests to validate the results.
# ###Your task:
# ###document:
# The problem statement, all variables and given/known data Two frats compete in a tug of war. Each team has 5 players of mass 80kg each. Each person is capable of pulling with 100N of force. The force of one team decays according to F(t) = F0(e^(-t/2)) and the other, F(t) =F0(e^(-t/1)). What is the position of the (very lightweight) rope as a function of time? 2. Relevant equations Fnet = ma 3. The attempt at a solution (m)(a) =F0(e^(-t/1)) - F0(e^(-t/2)) where F0 = 500 for both team. I am not sure if the mass is simply 5*80 or double that because of the two teams. Then I suppose you could integrate this and find x from the acceleration.
# ###important:
# 1. **Do not** reuse information from the original document for the new question.
# 2. The new question should be able to be answered **without information from the documentation**.
# 3. **Referring to the examples I provided**, the new questions you generate should be **different** from the documentation to ensure that the new questions are diverse, but at the same time **maintain some relevance**.
# 4. Ensure that the new question you generate is of **high quality**.
# 5. **Do not** provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# '''

# 有reason
# prompt = '''Generate a new question based on the provided document.
# Before I give you the document, I'll give you two examples:
# Example 1:
# ###document:
# I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
# [New Question]:
# How does the use of immersive virtual reality technology enhance the experience of exploring ancient ruins, and what are the potential ethical considerations when reconstructing historical sites for virtual tours?
# Example 2:
# ###document:
# Karen works at an animal shelter with 30 dogs, 28 cats, and 20 lizards. Every month, 50% of the dogs are adopted, 25% of the cats are adopted, and 20% of lizards are adopted. If the shelter takes in 13 new pets a month, how many pets are there after one month?
# [New Question]:
# A company sells 120 units of a product at $40 each. For every 20 units sold, the company offers a 5% discount on the price of those units. Additionally, the company incurs a $200 fixed cost and $10 per unit variable cost. What is the total profit after all sales?
# Example 3:
# ###document:
# function collectWithWildcard(test) { test.expect(4); var api_server = new Test_ApiServer(function handler(request, callback) { var url = request.url; switch (url) { case '/accounts?username=chariz*': let account = new Model_Account({ username: 'charizard' }); return void callback(null, [ account.redact() ]); default: let error = new Error('Invalid url: ' + url); return void callback(error); } }); var parameters = { username: 'chariz*' }; function handler(error, results) { test.equals(error, null); test.equals(results.length, 1); var account = results[0]; test.equals(account.get('username'), 'charizard'); test.equals(account.get('type'), Enum_AccountTypes.MEMBER); api_server.destroy(); test.done(); } Resource_Accounts.collect(parameters, handler); } module.exports = { collectWithWildcard };
# [New Question]:
# Write a function that takes a wildcard string as input to search for user accounts from an API, handles potential API errors, and ensures that the returned data matches the expected format (e.g., correct username and account type). Additionally, implement tests to validate the results.
# ###Your task:
# ###document:
# John prepares materials to build 12 towers that each require 2 square meters of plastic to construct plus 4 rounds of 40,000 drops of consecrated water. If John has just enough time to only prepare materials for 5 towers, calculate the total volume of drops of consecrated water he needs to have. New questions in Mathematics
# ###important:
# 1. Don't reuse information from the original document for new question.
# 2. New questions should be able to be answered without information from the documentation.
# 3. Following the example I provided, the new questions you generate should be quite different from the documentation to ensure that the new questions are diverse, but at the same time maintain some relevance.
# 4. You need to explain the relationship and diversity of the new question to the original document.
# 5. Ensure that the new question you generate is of high quality.
# 6. Don't provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# [Explanation]: Here is the explanation for the new question.
# '''



# 实验最终用prompt
# prompt = '''Generate a new question based on the provided document.
# Before I give you the document, I'll give you three examples:
# ###Examples:
# Example 1:
# ###document:
# I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
# [New Question]:
# How does the use of immersive virtual reality technology enhance the experience of exploring ancient ruins, and what are the potential ethical considerations when reconstructing historical sites for virtual tours?
# Example 2:
# ###document:
# tag:blogger.com,1999:blog-18922786044488938322014-10-04T19:53:09.407-07:00Little Bob's BlogJust a place for me to rant that is not on facebook!Bob Heathcote Android Market works only with devices with cell/data servicesHa, I was right: <br />.<br /><br /.<br /><br />That is why you need a cell service to have Android Market on your Android tablet. <br />OK, they could have Market on a device without a IMSI, but a lot of apps would not install and complain there is no IMSI to tie the license to.<br /><br />Source: <a href=\"\">Android Developers page</a> <br /><br /><br />More on <a href=\"\">IMSI</a>Bob Heathcote Laguna Seca race reportIt was a pretty good race, up until the oil filter failed on the leading car. Incredible<br /><br />Pretty good read <a href=\"\">here</a>Bob Heathcote 500 on the come back?<p class=\"MsoNormal\"? </p> <p class=\"MsoNormal\"><o:p> That said the race will probably be a total let down with a Penske/Helio runaway</o:p></p> <p class=\"MsoNormal\"><o:p> Discuss?</o:p></p>Bob Heathcote
# [New Question]:
# Write a Python script that checks for IMSI availability on an Android emulator, simulates app installation, and logs whether the app installation fails due to missing IMSI or other dependencies.
# Example 3:
# ###document:
# law relating the apparent contrast, Note 1 to entry: The formula is sometimes written where the exponent, Note 2 to entry: Taking into account the relationship between atmospheric transmissivity, Note 3 to entry: The contrast is taken to be the quotient of the difference between the luminance of the object and the luminance of the background, and the luminance of the background.  Note 4 to entry: This entry was numbered 845-11-22 in IEC 60050-845:1987.  Note 5 to entry: This entry was numbered 17-629 in CIE S 017:2011.
# [New Question]:
# If the luminance of an object is \( L_o = 250 \, \text{cd/m}^2 \), the luminance of the background is \( L_b = 100 \, \text{cd/m}^2 \), and atmospheric transmissivity reduces the object's luminance by \( t = 0.8 \), what is the apparent contrast? Use the formula: \[ C = \frac{(t \cdot L_o) - L_b}{L_b}. \].
# ###Your task:
# ###document:
# Solving Linear Systems by Graphing For this linear systems worksheet, 9th graders solve and complete 4 different problems by graphing.  First, they graph the first equation shown.  Then, students graph the second equation on the same coordinate system as the first.  They also find the solution and check the proposed ordered pair solution in both equations.
# ###important:
# 1. **Do not** reuse information from the original document for the new question.
# 2. The new question should be able to be answered **without information from the documentation**.
# 3. **Referring to the examples I provided**, the new questions you generate should be **different** from the documentation to ensure that the new questions are diverse, but at the same time **maintain some relevance**.
# 4. Ensure that the new question you generate is of **high quality**.
# 5. **Do not** provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# '''

# prompt = '''What do you think are the important attributes that make up the following document,  please change the attributes to generate a new question, where subtopic is an attribute that must exist and change.
# Before I give you the document, I'll give you two examples of documents and its important attributes:
# Example 1:
# ###document:
# Impact of network delays on Hyperledger Fabric. Blockchain has become one of the most attractive technologies for applications, with a large range of deployments such as production, economy, or banking. Under the hood, Blockchain technology is a type of distributed database that supports untrusted parties. In this paper we focus Hyperledger Fabric, the first blockchain in the market tailored for a private environment, allowing businesses to create a permissioned network. Hyperledger Fabric implements a PBFT consensus in order to maintain a non forking blockchain at the application level. We deployed this framework over an area network between France and Germany in order to evaluate its performance when potentially large network delays are observed. Overall we found that when network delay increases significantly (i.e. up to 3.5 seconds at network layer between two clouds), we observed that the blocks added to our blockchain had up to 134 seconds offset after 100 th block from one cloud to another. Thus by delaying block propagation, we demonstrated that Hyperledger Fabric does not provide sufficient consistency guaranties to be deployed in critical environments. Our work, is the fist to evidence the negative impact of network delays on a PBFT based blockchain.
# [Attributes]:
# Writing Style: Encouraging papers with different writing styles, such as technical, expository,
# theoretical, or empirical, can bring diversity to the presentation and appeal to a wider range of readers.
# Subtopics: Promoting papers that explore different subtopics within the broader topic can provide
# comprehensive coverage and delve into specific areas of interest.
# Techniques: Encouraging papers that employ different research methodologies, such as experimental,
# computational, or analytical, can bring diverse approaches to studying the topic.
# Data Sources: Promoting papers that utilize diverse data sources, such as surveys, simulations, 
# real-world datasets, or case studies, can offer different perspectives and insights into the topic.
# Interdisciplinary Perspectives: Encouraging papers that incorporate interdisciplinary perspectives,
# drawing insights from multiple fields or combining methodologies from different disciplines, can
# contribute to a richer understanding of the topic.
# [New Question]:
# How does the RBF-FD-inspired technique efficiently approximate definite integrals over the volume of a ball using arbitrarily scattered nodes without requiring uniformity?
# Example 2:
# ###document:
# Don't waste your time even just simply flipping through this magazine at the newstand. Trust me... there will be no worthwhile patterns (I mean... just look at what they choose to grace their cover!!!) I have stopped hoping for even a half way decent pattern to come along during the two years or so. I have seen litteraly only ONE knittable pattern in the last three years (December 2004 I believe it was... a lace cardigan with beaded trim). Even the articles are pointless. Save your time and money and get a subscription to Interweave Knits or Vogue Knitting. Or opt for a new magazine (not available at all newstands in the US) called Simply Knitting, which is imported from the UK. It amazes me that this magazine is even still in production with how horible it is!!
# [Attributes]:
# Product Type: Clearly mention the type of product you are reviewing, such as a smartphone, laptop, or fitness tracker. This helps readers understand the category and purpose of the product.
# Brand: Specify the brand of the product as it often influences quality, reputation, and customer support. Discuss the brand's overall credibility and whether it aligns with your expectations.
# User Experience: Evaluate the overall user experience of the product. Discuss its speed, accuracy, reliability, and efficiency in performing its intended tasks. Highlight any exceptional or lacking performance aspects.
# Quality and Durability: Assess the quality of the product, including the materials used, construction, and overall durability. Discuss whether it feels well-made, solid, and likely to withstand regular use over time.
# Features and Functionality: Describe the specific features and functions of the product. Highlight any unique or standout features that enhance its usability or set it apart from similar products in the market.
# [New Question]:
# How can Men's Health Magazine help men improve their overall lifestyle and address key health and wellness concerns?
# ###Your task:
# ###document:
# Isaac Newton was born in 1642 in a small English village, and from an early age, he was curious about the world around him. One day, while sitting under a tree, he watched an apple fall and wondered, Why does it always fall straight down? This simple question sparked a lifelong quest to understand the forces that govern the universe.
# At Cambridge University, Newton studied math and science, and he realized that the same force causing the apple to fall was also keeping the planets in orbit. This insight led him to develop the laws of motion and gravity, which revolutionized our understanding of the universe.
# Despite his groundbreaking discoveries, Newton remained humble, often acknowledging the work of those who came before him. His curiosity and determination changed the course of science, and his laws continue to shape our world today.
# ###important:
# The attributes before and after the change should be independent of each other.
# You need to explain the reason for the original attributes and the changed attributes.
# Ensure that the new question you generate is of high quality.
# Don't provide a solution or answer to the question.
# Your output should be as follows:
# [Attributes]: Here are the important attributes that make up the following document.
# [New Question]: Here is the new question.
# [Reason]: Reason for the original attributes and the changed attributes.
# '''

# prompt = '''Generate a new question based on the provided document.
# Before I give you the document, I'll give you three examples:
# ###Examples:
# Example 1:
# ###document:
# I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
# [New Question]:
# How does the use of immersive virtual reality technology enhance the experience of exploring ancient ruins, and what are the potential ethical considerations when reconstructing historical sites for virtual tours?
# Example 2:
# ###document:
# tag:blogger.com,1999:blog-18922786044488938322014-10-04T19:53:09.407-07:00Little Bob's BlogJust a place for me to rant that is not on facebook!Bob Heathcote Android Market works only with devices with cell/data servicesHa, I was right: <br />.<br /><br /.<br /><br />That is why you need a cell service to have Android Market on your Android tablet. <br />OK, they could have Market on a device without a IMSI, but a lot of apps would not install and complain there is no IMSI to tie the license to.<br /><br />Source: <a href=\"\">Android Developers page</a> <br /><br /><br />More on <a href=\"\">IMSI</a>Bob Heathcote Laguna Seca race reportIt was a pretty good race, up until the oil filter failed on the leading car. Incredible<br /><br />Pretty good read <a href=\"\">here</a>Bob Heathcote 500 on the come back?<p class=\"MsoNormal\"? </p> <p class=\"MsoNormal\"><o:p> That said the race will probably be a total let down with a Penske/Helio runaway</o:p></p> <p class=\"MsoNormal\"><o:p> Discuss?</o:p></p>Bob Heathcote
# [New Question]:
# Write a Python function that takes a list of integers and returns a dictionary mapping each unique integer to the count of its occurrences in the list. Ensure your function can handle empty lists and lists with negative numbers.
# Example 3:
# ###document:
# law relating the apparent contrast, Note 1 to entry: The formula is sometimes written where the exponent, Note 2 to entry: Taking into account the relationship between atmospheric transmissivity, Note 3 to entry: The contrast is taken to be the quotient of the difference between the luminance of the object and the luminance of the background, and the luminance of the background.  Note 4 to entry: This entry was numbered 845-11-22 in IEC 60050-845:1987.  Note 5 to entry: This entry was numbered 17-629 in CIE S 017:2011.
# [New Question]:
# If a car travels at a speed of \( v = 60 \, \text{km/h} \) for 2 hours, and then the speed is increased to \( v = 80 \, \text{km/h} \) for another 3 hours, what is the total distance traveled? Use the formula: \[ d = v \cdot t, \] where \( d \) is the distance, \( v \) is the velocity, and \( t \) is the time.
# ###Your task:
# ###document:
# Solving Linear Systems by Graphing For this linear systems worksheet, 9th graders solve and complete 4 different problems by graphing.  First, they graph the first equation shown.  Then, students graph the second equation on the same coordinate system as the first.  They also find the solution and check the proposed ordered pair solution in both equations.
# ###important:
# 1. **Do not** reuse information from the original document for the new question.
# 2. The new question should be able to be answered **without information from the documentation**.
# 3. **Referring to the examples I provided**, the new questions you generate should be **different** from the documentation to ensure that the new questions are diverse, but at the same time **maintain some relevance**.
# 4. Ensure that the new question you generate is of **high quality**.
# 5. **Do not** provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# '''

prompt = '''Generate a new question based on the provided document.
Before I give you the document, I'll give you three examples:
###Examples:
Example 1:
###document:
I got off the plane, grabbed my bags, I saw she on the other side of the door waiting for me, my legs were shaking, my heart was beating fast, breathing breathlessly, I walked up to her, I dropped his bags and gave the best hug of my life for the most perfect girl I've ever seen. I felt floating in the clouds. Tomorrow I'll be miles away but I'll still be able to feel her in my arms. (via divine-infection)
[New Question]:
How does the use of immersive virtual reality technology enhance the experience of exploring ancient ruins, and what are the potential ethical considerations when reconstructing historical sites for virtual tours?
Example 2:
###document:
tag:blogger.com,1999:blog-18922786044488938322014-10-04T19:53:09.407-07:00Little Bob's BlogJust a place for me to rant that is not on facebook!Bob Heathcote Android Market works only with devices with cell/data servicesHa, I was right: <br />.<br /><br /.<br /><br />That is why you need a cell service to have Android Market on your Android tablet. <br />OK, they could have Market on a device without a IMSI, but a lot of apps would not install and complain there is no IMSI to tie the license to.<br /><br />Source: <a href=\"\">Android Developers page</a> <br /><br /><br />More on <a href=\"\">IMSI</a>Bob Heathcote Laguna Seca race reportIt was a pretty good race, up until the oil filter failed on the leading car. Incredible<br /><br />Pretty good read <a href=\"\">here</a>Bob Heathcote 500 on the come back?<p class=\"MsoNormal\"? </p> <p class=\"MsoNormal\"><o:p> That said the race will probably be a total let down with a Penske/Helio runaway</o:p></p> <p class=\"MsoNormal\"><o:p> Discuss?</o:p></p>Bob Heathcote
[New Question]:
Write a Python script that parses an HTML blog post and extracts the relevant metadata, such as the date, title, and links. The script should handle any malformed HTML tags, such as unclosed or misplaced <br> or <p> tags, and ensure the data is correctly formatted for further processing.
Example 3:
###document:
law relating the apparent contrast, Note 1 to entry: The formula is sometimes written where the exponent, Note 2 to entry: Taking into account the relationship between atmospheric transmissivity, Note 3 to entry: The contrast is taken to be the quotient of the difference between the luminance of the object and the luminance of the background, and the luminance of the background.  Note 4 to entry: This entry was numbered 845-11-22 in IEC 60050-845:1987.  Note 5 to entry: This entry was numbered 17-629 in CIE S 017:2011.
[New Question]:
An object is placed in front of a lens with a focal length of \( f = 50 \, \text{cm} \). The object is positioned at a distance of \( d_o = 150 \, \text{cm} \) from the lens, and an external factor, such as the medium's refractive index, changes the object distance by a factor of \( r = 0.8 \). How would you calculate the apparent image distance \( d_i \) using the lens equation: \(\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}\).
###Your task:
###document:
The problem statement, all variables and given/known data Two frats compete in a tug of war. Each team has 5 players of mass 80kg each. Each person is capable of pulling with 100N of force. The force of one team decays according to F(t) = F0(e^(-t/2)) and the other, F(t) =F0(e^(-t/1)). What is the position of the (very lightweight) rope as a function of time? 2. Relevant equations Fnet = ma 3. The attempt at a solution (m)(a) =F0(e^(-t/1)) - F0(e^(-t/2)) where F0 = 500 for both team. I am not sure if the mass is simply 5*80 or double that because of the two teams. Then I suppose you could integrate this and find x from the acceleration.
###important:
1. Do not reuse information from the original document for the new question.
2. The new question should be able to be answered without information from the documentation.
3. Referring to the examples I provided, the new questions you generate should be different from the documentation to ensure that the new questions are diverse, but at the same time maintain some relevance.
4. Ensure that the new question you generate is of high quality.
5. Do not provide a solution or answer to the question.
Your output should be as follows:
[New Question]: Here is the new question.
'''
# Referring to the examples I provided, 
while True:
    conversation = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
                conversation,
                # return_dict=True,
                return_tensors="pt",
    )

    inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=5000, do_sample= False, temperature=0.5)# True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(result)
    print('test')

# prompt = '''Generate a new question.
# ###important:
# 1. Ensure that the new question you generate is of high quality.
# 2. Don't provide a solution or answer to the question.
# Your output should be as follows:
# [New Question]: Here is the new question.
# '''

# while True:
#     conversation = [{"role": "user", "content": prompt}]
#     inputs = tokenizer.apply_chat_template(
#                 conversation,
#                 # return_dict=True,
#                 return_tensors="pt",
#     )

#     inputs = inputs.to('cuda')
#     outputs = model.generate(inputs, max_new_tokens=5000, do_sample= True, temperature=0.7, top_p=0.9)# True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
#     result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
#     print(result)
#     print('test')