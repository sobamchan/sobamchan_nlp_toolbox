from slackclient import SlackClient
import os

class Slack(object):

    def __init__(self):
        pass

    @staticmethod
    def s_print(text, channel=None):
        if channel:
            slack_token = os.environ["SLACK_API_TOKEN"]
            sc = SlackClient(slack_token)
            sc.api_call(
              "chat.postMessage",
              channel='#'+channel,
              text=text
            )
        print(text)
