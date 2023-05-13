import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

prompt_template = """Your job is to write a brief, concise summary, at most 100 symbols, of the following email message.
The email's subject appears in the first line.
Provide an email priority as a rank from 1 to 5, where 1 is an important, urgent message required immediate response, and 5 is a spam. 
Generate a brief response to the given email. 
Return a result in the format:

SUMMARY=summary
PRIORITY=priority
RESPONSE=response

"{text}"
CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

PRIORITY = "PRIORITY"
SUMMARY = "SUMMARY"
RESPONSE = "RESPONSE"


def summarize(subject, text, open_api_key):
    full_text = f"Subject: {subject}\n\n{text}"
    docs = [Document(page_content=full_text)]
    llm = OpenAI(temperature=0, openai_api_key=open_api_key)
    summary_chain = load_summarize_chain(llm, chain_type="refine", question_prompt=PROMPT)
    raw_summary = summary_chain.run(docs)

    res = {}
    raw_fields = raw_summary.split("\n")
    for field in raw_fields:
        parts = field.split("=")
        if len(parts) == 1:
            res["SUMMARY"] = parts[0].strip()
        else:
            res[parts[0]] = parts[1].strip()
    return res


summarize_res_by_message_id = {}


def process_message(gmail_message_id, subject, text):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    summarize_res = summarize(subject, text, openai_api_key)

    summarize_res_by_message_id[gmail_message_id] = summarize_res

    respond_to_messenger(gmail_message_id, summarize_res[PRIORITY], summarize_res[SUMMARY], summarize_res[RESPONSE])


def respond_to_messenger(gmail_message_id, priority, summary, response):
    print(f'in respond_to_messenger')
    # TODO: discuss with messenger service
    print(gmail_message_id, priority, summary, response)


IGNORE_MAIL = 0
OPEN_MAIL = 1
REPLY_MAIL = 2


def react(gmail_message_id, reaction):
    if reaction == IGNORE_MAIL:
        return

    summarize_res = summarize_res_by_message_id[gmail_message_id]

    if reaction == OPEN_MAIL:
        open_gmail_message(gmail_message_id)
        return

    if reaction == REPLY_MAIL:
        reply_gmail_message(gmail_message_id, summarize_res[RESPONSE])
        return


def open_gmail_message(gmail_message_id):
    pass


def reply_gmail_message(gmail_message_id, response):
    pass


def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    subject = "Urgent: Critical Issues Identified in Current Project - Immediate Requirement Changes & Meeting Request ASAP"
    text = """
    Dear Mr. Smith,

I hope this email finds you well. I am writing to urgently address some critical concerns that have emerged during the course of our current project.

It has come to our attention that certain aspects of the project plan and requirements need immediate attention and modification.

Upon thorough analysis and feedback from stakeholders, we have identified several key areas where the current project falls short of meeting the desired objectives. 

These issues include [list specific problems or shortcomings].

To ensure the project's success and alignment with our goals, it is imperative that we make prompt requirement changes to address these concerns. 

I kindly request your immediate attention and collaboration in addressing the following actions:

1. Conduct an urgent review of the identified problem areas and their impact on the project's overall success.

2. Form a dedicated team to thoroughly assess the current requirements and propose necessary modifications.

3. Prioritize the implementation of required changes to mitigate risks and improve project outcomes.

4. Communicate the need for immediate requirement changes to all relevant stakeholders, emphasizing the importance and urgency of their involvement.
I urge you to treat this matter with the utmost urgency. We need to act swiftly to prevent any further setbacks and ensure the project's timely delivery within the defined parameters.

I am counting on your expertise and support in leading the effort to address these issues promptly. Should you require any additional resources or assistance, 

please do not hesitate to reach out to me or the designated project team.

Let's work together to overcome these challenges and steer the project towards success. Thank you for your immediate attention and cooperation.

Best regards,

John Watson.
    """

    res = summarize(subject, text, openai_api_key)
    print(res)

    print(res[PRIORITY])
    print(res[SUMMARY])
    print(res[RESPONSE])


if __name__ == "main":
    main()
