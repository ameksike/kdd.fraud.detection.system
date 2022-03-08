from services.SingletonMeta import SingletonMeta

class EdaService(metaclass=SingletonMeta):
    
    def getOutlierFields(self): 
        return [
            { "name": 'transaction_amount', "labels": ['small', 'medium', 'big'] },
            { "name": 'user_balance', "labels": ['small', 'medium', 'big'] }
        ]
    
    def getProperties(self):
        return [
            'tpv_name', 'billing_country', 'billing_user_country',
            'card_brand', 'card_sub_brand', 'email_domain',
            'ip_geo_code', 'ip_geo_region', 'ip_geo_timezone',
            'order_memo', 'tpp_error_reason',
            'user_country', 'user_verification_level', 'user_zip',
            'email_valid', 'ip_vpn',
            'phone_country', 'phone_valid',
            'user_category', 'user_agent_browser_family', 'user_agent_device_family',
            'user_agent_os_family', 'user_account_status',
            'transaction_type', 'transaction_currency',
            'tpv_card_dcc', 'tpv_alert_indicator', 'tpv_card3d_s2_partition',
            'tpv_card_aut3_d_secure2_method',
            'same_card_operations', 'same_card_operations_day',
            'same_card_operations_fraud',
            'same_card_shared_ip', 'same_card_shared_ip_day', 'same_card_shared_ip_fraud',
            'same_card_shared_user', 'same_card_shared_user_day', 'same_device_fingerprint',
            'same_device_fingerprint_fraud', 'same_ip_operations', 'same_ip_operations_day',
            'same_ip_operations_fraud', 'same_user_operations_day',
            'transaction_amount', 'user_balance',
            'fraud_state'
        ]
