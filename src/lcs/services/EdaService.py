from services.SingletonMeta import SingletonMeta

class EdaService(metaclass=SingletonMeta):
    def getProperties(self):
        return [
            'tpv name', 'billing country', 'billing user country',
            'card brand', 'card sub brand', 'email domain',
            'ip geo code', 'ip geo region', 'ip geo timezone',
            'order memo', 'tpp error reason',
            'user country', 'user verification level', 'user zip',
            'email valid', 'ip vpn',
            'phone country', 'phone valid',
            'user category', 'user agent browser family', 'user agent device family',
            'user agent os family', 'user account status',
            'transaction type', 'transaction currency',
            'tpv card dcc', 'tpv alert indicator', 'tpv card3d s2 partition',
            'tpv card aut3 d secure2 method',
            'same card operations', 'same card operations day',
            'same card operations fraud',
            'same card shared ip', 'same card shared ip day', 'same card shared ip fraud',
            'same card shared user', 'same card shared user day', 'same device fingerprint',
            'same device fingerprint fraud', 'same ip operations', 'same ip operations day',
            'same ip operations fraud', 'same user operations day',
            'transaction amount', 'user balance',
            'fraud state'
        ]
