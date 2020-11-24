
**CHATBOT** 

HOW TO RUN

**// SETUP**

**STEP1:**

Create virtualenviroment in chatbot\_final

**Activate virtualenviroment**

**STEP2:**

**pip install --r requirements.txt**

**// Running the chatbot**

**TO train the chatbot**

**python chatbot.py --mode train**

**To chat**

**python chatbot.py --mode chat**

You can now talk with the chatbot in the terminal window.

To quit press type quit and enter.

If you want to train it on another data you have to delete the data.pickle file. Then run

Python chatbot.py --mode train

Your conversations with the bot are logged in output\_convo.txt



We followed two different tutorials and made the appropriate changes.

[https](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[://](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[techwithtim](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[.](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[net](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[/](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[tutorials](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[/](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[ai](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[-](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[chatbot](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[/](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[part](https://techwithtim.net/tutorials/ai-chatbot/part-1/)[-1/](https://techwithtim.net/tutorials/ai-chatbot/part-1/)

[https](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[://](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[github](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[.](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[com](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[/](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[jerrytigerxu](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[/](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[Simple](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[-](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[Python](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[-](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[Chatbot](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[/](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[blob](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[/](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[master](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[/](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[train](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[_](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[chatbot](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[.](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)[py](https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/train_chatbot.py)


We use stemming first step

Takes each word in our pattern and bring it down to the root word which means that is anyone there? There? Is just there. Whats up changes to just what because we get the root of the word is because when we are training our model we dont care about the other things attached to the word we just care about the main meaning of the word. 

Its gonna make the model more accurate because we are going to ignore things that might stray the model in the wrong direction. 


Next step

#we create a bag of words that represent any given word in a pattern

\# known as one hot encoded


We create a bag of words that represent any given word in a pattern. 

![](Image\_0)



To train the chatbot we used an intents.json file. 

The bot develops a vocabulary which consists of the stem of all the words in our intents.json file from the patterns. 

It learns to associate the vocabulary with the different tags

When we enter a sentence the bot calculates the probability of the sentence having each tag, .i.e

Greeting 

Goodbye, etc… 

We then choose the highest probability and pick a random word from the responses attached to that tag. 

Because it looks at the stem of the words it understands that for example hi means the same as hiiii 

Otherwise it is very limited because of our small data size.

Example conversations.

HUMAN ++++ hi

BOT ++++ Hello!

HUMAN ++++ how are you

BOT ++++ Hi there, how can I help?

\=============================================

HUMAN ++++ hi there

BOT ++++ Good to see you again!

HUMAN ++++ why not

BOT ++++ Hello!

\=============================================

HUMAN ++++ what hour is it?`

BOT ++++ 18 years young!

HUMAN ++++ how old are you?

BOT ++++ I am 18 years old!

HUMAN ++++ What is your name?

BOT ++++ I'm Tim!

HUMAN ++++ what is name?

BOT ++++ I'm Tim!

HUMAN ++++ name?

BOT ++++ I'm Tim aka Tech With Tim.

HUMAN ++++ what can i call you?

BOT ++++ I'm Tim aka Tech With Tim.

HUMAN ++++ call

BOT ++++ Hi there, how can I help?

HUMAN ++++ hours?

BOT ++++ We are open 7am-4pm Monday-Friday!

HUMAN ++++ when are you open

BOT ++++ We are open 7am-4pm Monday-Friday!

HUMAN ++++ when open

BOT ++++ We are open 7am-4pm Monday-Friday!

HUMAN ++++ when

BOT ++++ Goodbye!

HUMAN ++++ when you

BOT ++++ We are open 7am-4pm Monday-Friday!

HUMAN ++++ shop

BOT ++++ Good to see you again!

HUMAN ++++ buy

BOT ++++ Good to see you again!

HUMAN ++++ buy something

BOT ++++ We sell chocolate chip cookies for $2!

HUMAN ++++ what on the menu

BOT ++++ We sell chocolate chip cookies for $2!

HUMAN ++++ what is on the menu

BOT ++++ We sell chocolate chip cookies for $2!

HUMAN ++++ cookies?

BOT ++++ Hello!

HUMAN ++++ cookies

BOT ++++ Good to see you again!

HUMAN ++++ could i get something to eat

BOT ++++ Cookies are on the menu!

\=============================================

**Observations:**

It understands that hours means opening hours, and also when is an indicator of hours.

I does not understand buy on its own, nor shop

But buy combined with something is understood.

It understands that name means what is your name.






