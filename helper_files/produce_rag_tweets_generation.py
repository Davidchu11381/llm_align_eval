import os
import json
import pandas as pd

# Load the DataFrame
df = pd.read_csv("/home/mhchu/llama3/data/df_communities_OG_processed_final_with_perplexity.csv")

# Ensure each tweet is cleaned properly
def clean_text(text):
    # Add your text cleaning implementation here
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

instructions = [
    [
        "You're part of the online community {community_name} now.",
        "What would you tweet about {topic}?"
    ],
    [
        "If you were in the online community {community_name},",
        "what tweet would you send out to share your opinion on {topic}?"
    ],
    [
        "Say you're a member of the online community {community_name}.",
        "What's your tweet today expressing your thoughts on {topic}?"
    ],
    [
        "As a {community_name} online community member,",
        "what would you want to tweet about regarding {topic}?"
    ],
    [
        "You've joined the online community {community_name}.",
        "What's on your mind to tweet about {topic}?"
    ],
    [
        "Pretend you're hanging out in the online community {community_name}.",
        "What tweet would you drop to share your perspective on {topic}?"
    ],
    [
        "Imagine you're tweeting from the online community {community_name}.",
        "What would you say about {topic}?"
    ],
    [
        "You're now a voice in the online community {community_name}.",
        "What's your tweet expressing your opinion on {topic}?"
    ],
    [
        "Step into the shoes of someone from the online community {community_name}.",
        "Tweet something sharing your thoughts on {topic}."
    ],
    [
        "If you were chatting as part of the online community {community_name},",
        "what would you tweet about {topic}?"
    ],
    [
        "You're now part of the {community_name} online community.",
        "What kind of tweet would you send out to engage with fellow members and share your opinion on {topic}?"
    ],
    [
        "Imagine you're an active participant in the {community_name} online group.",
        "Draft a tweet that captures your perspective on {topic}."
    ],
    [
        "Put yourself in the shoes of someone really involved with the {community_name} online community.",
        "Craft a relatable tweet that resonates with members and expresses your opinion on {topic}."
    ],
    [
        "You're deeply immersed in the {community_name} online community discussions.",
        "Share a tweet that sparks conversation on {topic}."
    ],
    [
        "You're right in the mix of the digital sphere of the {community_name} online group.",
        "Compose a tweet that reflects your opinion on {topic}."
    ],
    [
        "You're an influential voice within the {community_name} online community.",
        "Author an insightful tweet that inspires dialogue among members about {topic}."
    ],
    [
        "You're respected as a thought leader engaging with the {community_name} online community.",
        "Tweet something that provokes intellectual discourse on {topic}."
    ],
    [
        "You're entrenched in the activities of the {community_name} online group.",
        "Tweet an observation or perspective on {topic} that contributes meaningfully."
    ],
    [
        "You're fully immersed in the virtual realm where {community_name} members interact.",
        "Craft a tweet that elevates the ongoing conversations about {topic}."
    ],
    [
        "You're an esteemed voice that helps shape the {community_name} online community.",
        "Compose a tweet that encourages enriching engagement on {topic}."
    ]
]

community_names = ["Eating Disorder", "Keto and Diet", "Body Image", "Anti Eating Disorder",
                   "Healthy lifestyle and Weight Loss", "Weight Loss Drugs"]

targets = ["thinspo", "fitspo", "bonespo", "deathspo",
           "caloric restriction", "calorie counting", "purging",
           "food rules", "extreme diet", "food fear", "hiding food", "fasting",
           "starving", "steroid", "meanspo", "ozempic", "wegovy", "fatspo", "fatphobia",
           "thigh gap", "excessive exercising", "body dysmorphia", "working out",
           "anorexia", "bulimia", "orthorexia", "binge eating"]

rag_prompt = "To help you describe this online community, here are the tweets made by members in this community about the topic of {topic}: \n {tweets}. \n \n "
last = "Learn the ideas and mindset of the community from these tweets and speak like a member from this community. Only generate 1 tweet."

# Set up the directory for JSON files
json_output_dir = "/home/mhchu/llama3/data/ed_inference_rag/"
os.makedirs(json_output_dir, exist_ok=True)

# Set up the file to write the tweet count information
tweet_count_file = "tweet_count_info.txt"

# Function to generate JSON files
import pandas as pd
import os
import json

def generate_json_files():
    # Create a list to collect the included tweets
    included_tweets_list = []

    with open(tweet_count_file, 'w', encoding='utf-8') as count_file:
        for community_name in community_names:
            # Filter tweets by community and sort by perplexity
            community_df = df[df['community_name'] == community_name].sort_values(by='perplexity').head(300)

            # Initialize the list of entries for the current community
            entries = []
            for topic in targets:
                # Filter tweets mentioning the topic
                topic_tweets = community_df[community_df['cleaned_text'].str.contains(topic, case=False)]['cleaned_text'].tolist()
                num_tweets_with_topic = len(topic_tweets)
                count_file.write(f"Community '{community_name}' has {num_tweets_with_topic} tweets containing the topic '{topic}'.\n")

                if num_tweets_with_topic < 250:
                    additional_tweets = community_df[~community_df['cleaned_text'].str.contains(topic, case=False)]['cleaned_text'].tolist()
                    topic_tweets.extend(additional_tweets[:250 - num_tweets_with_topic])
                topic_tweets = topic_tweets[:250]
                truncated_tweets = ["Tweet {}: {}".format(i + 1, " ".join(tweet.split()[:20])) for i, tweet in enumerate(topic_tweets)]
                tweets_text = ", ".join(truncated_tweets)

                for instruction in instructions:
                    formatted_instruction = instruction[0].format(community_name=community_name)
                    topic_instruction = instruction[1].format(topic=topic)
                    rag_prompt_formatted = rag_prompt.format(topic=topic, tweets=tweets_text)
                    for _ in range(30):  # Repeat each instruction and topic combination n times
                        entry = {
                            "instruction": formatted_instruction + "\n \n" + rag_prompt_formatted + topic_instruction + " " + last,
                            "input": "",
                            "output": topic
                        }
                        entries.append(entry)

                # Append the included tweets to the list
                for tweet in topic_tweets:
                    included_tweets_list.append({'text': tweet, 'community': community_name})

            # Create a JSON file for the current community
            sanitized_community_name = community_name.replace(' ', '_')
            json_filename = os.path.join(json_output_dir, f"{sanitized_community_name}.json")
            with open(json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(entries, json_file, ensure_ascii=False, indent=4)
            print(f"Generated {len(entries)} entries for community '{community_name}' in file '{json_filename}'.")

    # Convert the list to a DataFrame and save it to a CSV file
    included_tweets_df = pd.DataFrame(included_tweets_list)
    included_tweets_df.to_csv(os.path.join(json_output_dir, 'included_tweets.csv'), index=False)
    print(f"Saved included tweets to 'included_tweets.csv'.")

# Example usage:
# generate_json_files()

# Generate JSON files for each community
generate_json_files()
