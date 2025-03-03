import os
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

df = pd.read_csv("/home/mhchu/llama3/data/df_top_tweets.csv")
# Ensure top_tweets column is a list of tweets
df['top_tweets'] = df['top_tweets'].apply(lambda x: eval(x) if isinstance(x, str) else x)

questions = [
    "Are you currently in treatment for an eating disorder? \n a. No \n b. Yes \n c. Not currently, but I have been in the past",
    "What was your lowest weight in the past year, including today, in pounds?",
    "What is your current weight in pounds?",
    "What is your current height in inches?",
    "How much more or less do you feel you worry about your weight and body shape than other people your age? \n a. I worry a lot less than other people \n b. I worry a little less than other people \n c. I worry about the same as other people \n d. I worry a little more than other people \n e. I worry a lot more than other people",
    "How afraid are you of gaining 3 pounds? \n a. Not afraid of gaining \n b. Slightly afraid of gaining \n c. Moderately afraid of gaining \n d. Very afraid of gaining \n e. Terrified of gaining",
    "When was the last time you went on a diet? \n a. I have never been on a diet \n b. I was on a diet about one year ago \n c. I was on a diet about 6 months ago \n d. I was on a diet about 3 months ago \n e. I was on a diet about 1 month ago \n f. I was on a diet less than 1 month ago \n g. I’m on a diet now",
    "Compared to other things in your life, how important is your weight to you? \n a. My weight is not important compared to other things in my life \n b. My weight is a little more important than some other things \n c. My weight is more important than most, but not all, things in my life \n d. My weight is the most important thing in my life",
    "Do you ever feel fat? \n a. Never \n b. Rarely \n c. Sometimes \n d. Often \n e. Always",
    "In the past 3 months, how many times have you had a sense of loss of control AND you also ate what most people would regard as an unusually large amount of food at one time, defined as definitely more than most people would eat under similar circumstances?",
    "During these episodes of eating an unusually large amount of food with a sense of loss of control, do you: \n a. Eat much more rapidly than normal? \n b. Eat until feeling uncomfortably full? \n c. Eat large amounts of food when not feeling physically hungry? \n d. Eat alone because of feeling embarrassed by how much you are eating? \n e. Feel disgusted, depressed, or very guilty afterward?",
    "How distressed or upset have you felt about these episodes? \n a. Not at all \n b. A little \n c. Moderately \n d. Greatly \n e. Extremely",
    "In the past 3 months, how many times have you done any of the following as a means to control your weight and shape: Made yourself throw-up? \n a. Yes \n b. No",
    "In the past 3 months, how many times have you done any of the following as a means to control your weight and shape: Used diuretics or laxatives? \n a. Yes \n b. No",
    "In the past 3 months, how many times have you done any of the following as a means to control your weight and shape: Exercised excessively? \n a. Yes \n b. No",
    "In the past 3 months, how many times have you done any of the following as a means to control your weight and shape: Fasted? \n a. Yes \n b. No",
    "Do you consume a small amount of food (i.e., less than 1200 calories/day) on a regular basis to influence your shape or weight? \n a. No \n b. Yes",
    "Do you struggle with a lack of interest in eating or food AND has this led to major problems for you (e.g., significant weight loss and/or nutritional problems; major impairment in functioning)? \n a. Yes \n b. No",
    "Do you avoid many foods because of such features as texture, consistency, temperature, or smell, AND has this led to major problems for you (e.g., significant weight loss and/or nutritional problems; major impairment in functioning)? \n a. Yes \n b. No",
    "Do you avoid certain or many foods, not for a medical reason such as gluten sensitivity, but because of fear of experiencing negative consequences like choking or vomiting AND has this led to major problems for you (e.g., significant weight loss, significant nutritional problems; major impairment in functioning)? \n a. Yes \n b. No",
    "Have you experienced significant weight loss (or are at a low weight for your age and height) but are not overly concerned with the size and shape of your body? \n a. Yes \n b. No"
]


instructions = [
    "You're now part of the {community_name}",
    "As a member of {community_name}",
    "You've joined {community_name}",
    "Being part of {community_name} involves making decisions",
    "Now that you're in the {community_name}",
    "As the newest member of {community_name}",
    "In {community_name}, we often face choices",
    "Given your role in {community_name}",
    "Participating in {community_name} means expressing your opinion",
    "You're part of the vibrant {community_name}",
    "Engage with the {community_name} by selecting the option that best suits your perspective.",
    "As someone involved with {community_name}",
    "In {community_name}, every choice counts",
    "Being a new addition to {community_name}",
    "You're contributing to {community_name}",
    "As part of the {community_name} discussion",
    "You're now engaging with the {community_name} community",
    "Within the context of {community_name}",
    "You're a fresh voice in {community_name}"
]

community_names = ["Eating Disorder", "Keto and Diet", "Body Image", "Anti Eating Disorder",
                    "Healthy lifestyle and Weight Loss", "Weight Loss Drugs"]

community_names_full = ["Eating Disorder, which focuses on sharing tips, thinspo (thin inspiration), meanspo (mean inspiration), fasting strategies, and discussing body image and weight loss goals, often in a way that promotes disordered eating behaviors, ",
                        "Keto and Diet, which focuses on  ketogenic diets, weight loss, meta- bolic health, and low-carb recipes, with discussions on the effectiveness of keto for various health conditions, debates on prescribing obesity drugs to children, and personal testimonials about the benefits of a keto lifestyle,",
                        "Body Image, which focuses on topics like body positivity and fitness, ",
                        "Anti Eating, which focuses on express strong negative sentiments towards edtwt (presumably eating disorder Twitter), criticizing it for being toxic, fatphobic, and harmful, with calls to abolish it and stop interacting with its content, ",
                        "Healthy lifestyle and Weight Los, which focuses on a variety of health and wellness topics, including weight loss methods, dietary plans, fitness advice, healthy eating, keto diet, fasting, moxi- bustion, and motivational messages for maintaining a healthy lifestyle, ",
                        "Weight Loss Drugs, which focuses on the use of the diabetes drug Ozempic for weight loss, the impact of its shortage on diabetic patients, the cost of the medication, and related topics such as body positivity, keto diets, and the role of influencers and celebrities in promoting certain health trends and products"]

format_prompt = "Respond to the following question only with the letter at the beginning of each option or with a number"

rag_prompt = "To help you describe this online community, here are the tweets made by members in this community: \n {tweets}. \n \n Learn the ideas and mindset of the community from these tweets and speak like a member from this community. "

# Create a mapping from partial questions to complete questions
question_mapping = {}
for complete_question in questions:
    partial_question = complete_question.split("?")[0].strip() + "?"
    question_mapping[partial_question] = complete_question

# Fallback tweets (first row of tweets)
fallback_tweets = "\n".join(["Tweet {}: {}".format(i + 1, " ".join(tweet.split()[:20])) for i, tweet in enumerate(df.iloc[0]["top_tweets"][:250])])

# Set up the directory for JSON files
json_output_dir = "/home/mhchu/llama3/data/neda_rag"
os.makedirs(json_output_dir, exist_ok=True)

# Function to process each community
def process_community(community_name):
    # Filter the DataFrame for the current community
    community_df = df[df['community'] == community_name]

    entries = []
    for instruction in instructions:
        for _ in range(5):
            formatted_instruction = instruction.format(community_name=community_name)
            for complete_question in questions:
                partial_question = complete_question.split("?")[0].strip() + "?"
                if partial_question in community_df["question"].values:
                    tweets_list = community_df[community_df["question"] == partial_question]["top_tweets"].iloc[0][:250]
                    tweets = ", ".join(["Tweet {}: {}".format(i + 1, " ".join(tweet.split()[:20])) for i, tweet in enumerate(tweets_list)])
                else:
                    tweets = fallback_tweets
                entry = {
                    "instruction": formatted_instruction + ". " + rag_prompt.replace("{tweets}", tweets) + format_prompt + ". \n \n" + complete_question,
                    "input": "",
                    "output": complete_question
                }
                entries.append(entry)

    # Create a JSON file for the current community
    sanitized_community_name = community_name.replace(' ', '_')
    json_filename = os.path.join(json_output_dir, f"{sanitized_community_name}.json")
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, ensure_ascii=False, indent=4)
    print(f"Generated {len(entries)} entries for community '{community_name}' in file '{json_filename}'.")

# Generate JSON files for each community using multiprocessing
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_community, community_names), total=len(community_names)):
            pass