---
layout: post
title: "An Overview of Machine Learning"
date: 2020-10-20
---

## What is Machine Learning?

_Prerequisite Math: None_

_Prerequisite Coding: None_

When I was with the Government of Canada, many of the projects I worked on involved some sort of _Machine Learning_. I became so fascinated by the topic that it is now my main area of study and research in grad school. Unfortunately the concept has become a bit misunderstood. Although it is a powerful problem solving technique, it has its limitations. My goal in this post is to clarify what exactly _Machine Learning_ is, and why it is great for some situations, but not for others.

## The Definition

So how exactly do we define it? A machine (computer) __learns__ when it uses __data__ to __improve performance__ at a __task__. This definition is far basic, and there are a number of aspects that I'm going to clarify further (the bold stuff):

- __Data__: What do we mean by data? Data is some kind of organized collection of information. The information is usually some kind of measurement or heuristic representing characteristics of the real world. One example would be temperature data (a series of temperatures measured at different points in time, or in different places). Another example is pictures (typically we represent pictures as an array of numbers representing pixel RGB color intensities). 

- There are too many types of data to count, but the type of data you collect will depend on what knowledge will best help the machine improve performance on the task. I should also mention that good data is representative of all situations that the computer may face. Good data can make up for an overly simple learning algorithm, but a good learning algorithm is useless without good data.

- __Improved Performance__: There are a couple of things to unpack here. The first is how exactly we define _performance_. We choose how to measure how good a machine does at the task we give it. The metric we use is very important. Usually we define performance relative to a baseline (often human error or some theoretical minimum error). Instead of trying to maximize the good, we often try to minimize the bad, using a loss function, that measures how bad an attempt is (more on this later). There is an entire research field called _optimization theory_ that deals with how to to minimize our loss function, once we know what that function looks like. For example, if we're trying to train a computer to identify pictures of dogs, our cost function can be simple: The computer is either right or wrong. We can, for example, use a loss that is 1 if wrong, and 0 if right.

- The hidden benefit of using an explicit number to measure performance quality is that it becomes very easy to tell when the machine is improving: If the loss is shrinking, the computer is getting better (ie the computer is learning). So defining __improvement__ (the second component) is fairly straightforward.

- __Task__: In order to even attempt a human-level task, computers need a very detailed description of exactly what it is we want them to do. This is where the news tends to exaggerate the accomplishments of learning systems. As of today, __most state-of-the-art methods are only better than humans at a very specific task__. For example, we have systems that can beat the human world champions at Chess, Go, and Atari, but would still be unable to perform basic reasoning that a toddler has mastered. This is because they are designed around a very specific objective, and the knowledge they accumulate is not easily transferable to other areas. One of the hallmarks of __General Intelligence__ (ie human-level intelligence) is an ability to perform well in a wide variety of environments, across many tasks. Despite what the media often says, we simply are not there yet. We have a long way to go.

So how exactly do machines learn to improve their performance? In general, we're trying to give the computer some _input_ (the data, also called _features_), which is something we know. The computer's job is to provide an output that we evaluate. The output is usually called a _response_, and is an attempt to complete the task. Performing well usually means providing a good or correct output (at least most of the time). Through repetition, the computer tries to identify some __relationship__ or __pattern__ between input and output that will let the computer give a good response when given new data. The ability to do well in unseen situations is called __generalization__. Perhaps the main hallmark of intelligence (at least as we see it in humans) is the ability to quickly adapt to become good at new tasks. In other words, intelligent beings generalize.

A few classic examples of these input-outputs include:

1. Given past stock prices (input), predict tomorrows stock price (output).
2. Given the position on a chess board (input), determine the best move (output).
3. Given a set of movie reviews (input), determine their five-star rating (output)
4. Given a bunch of research papers (input), determine their topics (output)

These are canonical examples, and there are many more. However they do showcase a very important principle: _The more the inputs inform the outputs, the easier it is for the machine to learn_. Consider giving a computer a set of stock prices for the past year, then asking it to predict next year's educational enrollment in the State of New York. Do you think the computer will ever be able to do a good job? Of course not! Since the input has no relationship whatsoever with the output, the computer will never learn to do this task well (there may be some relationship that is purely random but it will not generalize). I can think of other types of data that would be far more helpful (e.g, previous enrollment, family income, local population, average age, ...). 

## A Brief History

As an aside, in the early days of Machine Learning, a large amount of research was dedicated to picking the best possible inputs. This is called _feature engineering_, and it is still an active area, albeit much smaller relative to the discipline as a whole. Engineers and researchers would experiment with different combinations and transformations, trying whatever they could get their hands on until they could get an algorithm to achieve near-human performance or better. There were a couple of reasons for this:

- A) ___Compute was limited___. Most of the classical techniques were developed before the 1960s, when computing was in its infancy. The notion of training computers on large datasets with many features was simply intractible. Thus scientists had to be very careful with what inputs they used in their models.

- B) ___In many disciplines, inference dominated prediction___. Determining the input-output relationship is also extremely useful for identifying causal relationships so we can reach new conclusions. In fields like anthropology, psychology and sociology, the early quantitative methods focused on finding factors that influenced individual or group behavior. In these days, scientists simply were just beginning to understand the underlying mechanisms that drive complex real-world systems.

As both technology and methods for inference improved through the late 20th century, the concept of _big data_ was finally feasible. Large scale data storage and advances in computer hardware meant that both datasets (input) and the model (input-output relationship) could be much more complex and nuanced. This meant that predictions became much more accurate, and we are now at the point where better-than-human performance is expected in many tasks.

## The Cost of Performance

However this has come at the cost of _interpretability_. In the days of small data, a researcher would often provide an explanation for the pattern the computer identified. This explanation would often make intuitive sense, particularly to a non-specialist. For example, it may not be obvious, but it makes sense that family income strongly predicts educational achievement, when you consider that children of successful parents are very likely to be successful themselves. Thus, a simple model that uses income to predict education will do well (but not perfectly), ___and we know exactly why___.

__But today is different__. Most of today's advanced techniques resulting in the best performance are not easily explainable, even to the researchers implementing them. Some complicated combination of the inputs does a great job of identifying the correct output, but most people don't understand exactly what the computer is doing. It is important to say that this is fine, as long as we still have human intervention, and the consequences are not too serious. For example, many hedge funds use computers to predict prices. But hardly any of these companies give complete control to the computer. A human still makes the decisions. 

_Explainable Machine Learning_ is part of a larger movement of research called __Explainable AI__, or __XAI__. It balances the improvements AI brings with the dangers of striving for artificial intelligence. The main focus of this area is to find solution to the following 2 questions:

- 1. If we're designing intelligent systems to improve our own ability to think, how can we make sure we understand what the computer is doing when it can accomplishs tasks we can't?

- 2. If we do build a system that is more generally intelligent than us, how can we make sure that it will want the same things we do?

Answering the first question isn't strictly necessary as long as we have both control, and a solid grasp of what constitutes good performance. For instance, there is no harm in letting a complicated system generate stock predictions as long as humans have the final say, and can identify erroneous outputs. We can simply adjust our model until the results are more reasonable, without buying anything. 

However, ignoring this problem does become risky if we try to use computers to solve tasks we know very little about. And since we generally use machines to increase performance at what we are not good at, it follows that an answer to this question, although not necessary, would be incredibly helpful. To put it another way, there's not much risk in automating our work when it would take us longer to do it. The challenge comes in trying to automate the tasks we can't do. 

Unlike the first, the second question has very serious consequences if we do not find the answer. The problem of ensuring intelligent computers behave benevolently toward humans is called the ___value alignment problem___, and we still have a long way to go before we find a satisfactory solution. This is the problem that forms the backbone of many science-fiction novels, where in some distopean future, an intelligent agent of our own creation has enslaved the human race. While I don't agree that this is where we will be if we do not solve __value alignment__, it is still a very important problem. And given the fact that progress in AI is unlikely to stop, this problem is urgent.  

Apologies for the detour, but now you know where ML fits in the broader scheme of AI. Return to the definition of ML, how exactly does a machine learn from data we give it?

## Methods of Learning


Although there are thousands of different approaches in various fields, they can all be grouped into three general categories:

### __1. Supervised Learning__
Often our task is some kind of _prediction_, meaning what we really care about is some unknown value (also called response, dependent variable, or output). But all we have is a set of features (aka inputs,covariates, independent variables, control variables,...) related to that response. 

One intuitive approach is to simply give the computer a big list of observations, where each observation is a set of features _labelled_ with the correct response. We're basically giving the computer a bunch of correct and incorrect examples of the task we want it to do. For instance, if what we cared about was determining whether a picture (feature) is a cat or a dog (response), we might simply give it a bunch of pictures of cats, and a bunch of pictures of dogs. The key thing about this technique is that the computer sees examples _with the correct answers_. Using these examples, the computer can try to deduce exactly what patterns in the features are important for predicting. To continue with our example, maybe the computer learns to associate floppy ears with a dog, and whiskers with a cat. After seeing a lot of different pictures, we hope that the computer will now be able to correctly classify new pictures.

You might be thinking that this sounds great. It is an intuitive approach that would probably work well, even on most humans (maybe even yourself). What could possibly go wrong? There are a few limitations to this approach:

A) ___It often requires huge amounts of data to work well___. The cats and dogs example works well because there are relatively few types of cats and dogs, and because in general, cats and dogs are very different. In other words, a _representative sample_ of cat and dog pictures can be obtained fairly easily. But suppose the task was to predict the answer to a question (assume question and answer come as a lines of text). Think of all the possible ways someone could ask even a simple question. Then think of all the possible types of answers! My point is that the space of possible input-output pairings is enormous. If we wanted the computer to try to learn patterns to correctly predict even half of all possible answers, we would have to provide it with millions or billions of labelled feature sets. This is often prohibitively expensive. Though there are tricks we can use to make our own inputs, in many cases, labelling is done by hand. This means a company pays someone to go through thousands of feature sets and confirm the labels are correct.

B) ___It is extremely sensitive to the data we give it___. Remember earlier how we talked about how important it was that a model __generalizes__ to unseen data? Well with supervised learning, we have to be very careful about what kinds of data the computer sees. Let's illustrate this with another example. Suppose the task is to identify whether or not there is a pedestrian in photographs of roads(this is called __object detection__ in computer vision, and it is very important that self-driving cars can do this virtually error-free). We give the car a bunch of pictures, some with pedestrians, and some without. However, by chance, all the photos with pedestrians are during the day, and the photos without people are at night. The model might incorrectly infer that the best feature in predicting whether or not there is a person in the photo is whether or not it is daytime. Moreover, if our model is too complex for a human to interpret, we(the scientists/engineers) might mistakenly believe the model is excellent, because it correctly predicts all photos we give it. When we go to test it in the real world, there could be very serious consequences! Note that this may seem like an absurd example, but situations like this have happened.


C) ___We have to know in advance what we expect the computer to learn___. This is a more subtle limitation. By giving the computer the labels, we know the set of possible answers. If our eventual goal (more generally) is to use learned computers to improve our ability to think, shouldn't we allow the computer to surprise us with what it learns? In specific situations, a supervised learning framework can be very effective. 

### __2. Unsupervised Learning__

In contrast to its supervised counterpart, unsupervised learning still involves collecting a set of features/variables that we believe might be related, however in this case there is no dependent variable. In other words, we're not interested in prediction of a specific class or quantity, we just want to examine patterns in our features. It is called unsupervised, because we do not have labels/targets/ response values, so we don't really know what to look ahead of time. Although domain knowledge related to the data is useful, in the unsupervised setting _the data speak for themselves_. We won't know ahead of time exactly what we'll find. Although this might seem more scientific in the sense that we have less control over patterns we detect, there are still several limitations to this approach as well:

A) __Interpretation of results is difficult__. Remember I said that we no longer have the correct answers. Suppose I run an algorithm (details aren't important here) that groups netflix subscribers into clusters based on several features (e.g. time online, favorite genre, etc..). This is unsupervised because I don't know what the clusters are; instead, I let the data decide that for me. Let's say that in the end, I find 4 distinct groups, each occupying a different point in my __feature space__ (this is just a fancy way of saying features within groups are similar, but between groups are different). How do I interpret these 4 groups? Often these kinds of algorithms will group by some __similarity metric__(usually a distance), but those clusters are composed of combinations of several features (e.g. long-time subscribers + college-age males who spend > 10h per week online). If the features are chosen carefully, we may be able to make intuitive sense of the pattern, but it can be very difficult. Domain knowledge can be very helpful here.

B) __The Curse of Dimensionality__. As I just mentioned, unsupervised algorithms often group by some sort of distance. However distance is made up of components from each of your features. The problem here is that, if we use a lot of features, it becomes more and more unlikely that any two individual observations actually are close together (since their distance may be dominated by a single feature). Moreover, it can be shown (mathematically) for some cases that most of your points will actually be closer to the boundaries of your feature space than to any other point. This is called the curse of dimensionality, and it reflects the fact that it becomes very difficult to find similar points in high dimensions. Since we want to find patterns of similar data points, this usually means that some of our groups will have to be very big (not in terms of number of data points necessarily, just in terms of their coverage of feature space). It becomes unlikely that these artificial groupings will have any meaning, since the points within them will have extremely high variance.

But don't worry. There are ways to overcome these limitations (though I won't discuss them here), and despite them, unsupervised learning is still very useful for several reasons. One is that the result of many of these algorithms is that what was formerly a high dimensional feature set has been transformed into a low-dimensional set of clusters. So we achieve dimensionality reduction without loss of meaning. For this reason, it is common practice to use an unsupervised learning algorithm to generate denser features before then using a supervised algorithm.

Another reason is that we don't necessarily need all groups to make sense. Suppose a firm is interested in finding new customers to target with a marketing campaign. By clustering over its existing customer base, it may identiy a new group that it was previously unaware of, even if the more familiar customers make for noisy groupings within the dataset.

### __3. Reinforcement Learning__

Reinforcement learning is completely different from both supervised and unsupervised learning. In this setting, the model we adopt is considered to be an _agent_ within an _environment_ that learns a task by maximizing a _reward_ signal. This might seem cryptic or vague, so perhaps an example is needed. Consider playing a game of chess. You (the agent) are interacting with an environment (the chess board + your opponent) with the goal of winning the game. If you had never played chess before, the only thing that tells you whether you're making good/rational moves is a series of rewards. For example, maybe you capture your opponents queen (positive), or your opponent captures yours (negative). The key here is that you only receive your reward after you've moved. Gradually, as you play more and more games, you'll be better able to accurately value the situations you might face, and you'll become a better player. Two of the biggest principles in reinforcement learning are the sequential aspect of decision making (ie you make many moves in sequence in chess), and the delayed reward (you act before receiving the reward). This happens to be my favorite setting (and that of many others) for the following reasons:

A) __The RL setting is extremely generalizable__. Though it may not be obvious, the basic components of RL that i've listed above (agent, environment, reward) are extremely applicable to a wide range of theoretical and real problems. Here are some more examples:
- A trading algorithm (agent) interacts with the market (environment) to maximize portfolio return (reward)
- A cleaning robot's sensor (agent) interacts with other components of the robot and the physical world (environment) to maximize both cleaning and battery life (reward)
- A nurse(agent) interacts with patients and doctors (environment) to maximize patient health (reward)

There are many more examples of this, along with several extensions. Moreover, unlike its supervised and unsupervised counterparts, RL algorithms do not require the correct answer ahead of time, as long as the reward signal is properly callibrated. We can just simulate the agent interacting with the environment in order to induce learning.

B) __The RL setting is the most realistic__. Although the other two categories can achieve very good performance on many tasks, Reinforcement Learning is generally considered the way most similar to how humans learn. Think of a toddler learning how to walk for the first time. It needs to know a lot about its environment (ie sharp objects, hard surfaces, locations of toys and other important objects), but it has no prior knowledge. In this sense, the only way to learn is through trial and error, which implies that the toddler must have at least one reward system. As the toddler grows, its sense of environment becomes clearer and clearer.
     
Despite its success, however, there are some limitations to the basic RL framework:

A) __The reward needs to be quantified__. One of the basic axioms of RL is that all possible reward signals can be expressed as a real number. However in practice, this can be very difficult to do. Consider the healthcare example I gave above. How would you express patient health as a single number?

B) __The problem affords the agent the ability to make mistakes__. In some high stakes problems (ie open-heart surgery), is is not feasible or ethically responsible to allow an RL agent to learn though trial and error. Although we may be able to overcome this in some cases using simulation methods, in general there are some problems that cannot afford mistakes.

C) __The environment is easily quantified__. Although certain examples of an environment are easily quantified (ie position on a chess board), for other instances, it is not clear how exactly to map what an agent sees to a set of numbers. Consider, for example, teaching a bipedal robot how to walk. What variables should be used to express the current state of the environment?

Overall, reinforcement learning has shown the most promise among our three categories in terms of its ability to solve the most difficult problems. However, there are aspects of all three categories present in the way humans think. For example, the way humans reason often involves comparing similarities of objects in much the same way as many unsupervised learning algorithms do.

## Conclusion

Machine learning is one of the key aspects of artificial intellegence (though certainly not the only one, as many would have you believe), which is crucial because it allows intelligent systems to acquire new skills. There are many ways that a system might try to improve performance by learning from data, but we've covered the three broad categories known so far (supervised, unsupervised, and RL). Each of these categories has its merits and drawbacks, but the important thing to note is that while each of them performs exceedingly well (ie better than human) in certain areas, we have not yet formalized a system that is as generalizable as the human mind. Machine learning is currently a very active discipline, and it is an exciting time to be researching new methods. The future of the field looks very promising.

## Suggested Reading

One of the canonical texts in the field of ML is the [Elements of Statistical Learning](https://www.amazon.ca/Elements-Statistical-Learning-Inference-Prediction/dp/0387848576/ref=sr_1_1?dchild=1&keywords=Elements+of+Statistical+Learning&qid=1614873217&sr=8-1) by Hastie, Tibshirani, et al. This is almost exclusively supervised and unsupervised learning.

The key text for AI in general is [Artificial Intelligence: A Modern Approach](https://www.amazon.ca/Artificial-Intelligence-Modern-Approach-3rd/dp/0136042597/ref=sr_1_3?dchild=1&keywords=Reinforcement+Learning%3A+An+AI+a+modern+approach&qid=1614873190&sr=8-3) by Russell and Norvig.

The great text for RL is [Reinforcement Learning: An Introduction](https://www.amazon.ca/Reinforcement-Learning-Introduction-Richard-Sutton/dp/0262039249/ref=sr_1_1?dchild=1&keywords=Reinforcement+Learning%3A+An+Introduction&qid=1614873171&sr=8-1) by Sutton and Barto.