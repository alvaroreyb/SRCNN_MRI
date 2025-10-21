import telegram
__all__=["TelegramResults"]

def TelegramResults(text):
    bot = telegram.Bot("")
    bot.send_message(chat_id = ,text = text)