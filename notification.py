class Notifier:
    def send_email(self, to: str, subject: str, body: str):
        pass
    def send_sms(self, to: str, message: str):
        pass
    def send_push(self, user_id: str, message: str):
        pass
