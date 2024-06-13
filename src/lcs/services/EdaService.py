from numpy import float64
from services.SingletonMeta import SingletonMeta

class EdaService(metaclass=SingletonMeta):
    
    def getTypeFields(self): 
        # get data type from fields, the data types must be defined from an EDA analysis and not automatically to avoid errors in the classifier when they do not transfer values 
        return {
            'user_verification_level': str, 
            'email_valid': str, 
            'ip_vpn': str,
            'phone_valid': str,
            "tpv_name": str,
            "tpv_name": str,
            "billing_country": str,
            "billing_user_country": str,
            "card_brand": str,
            "card_sub_brand": str,
            "email_domain": str,
            "fraud_state": str,
            "ip_geo_code": str,
            "ip_geo_region": str,
            "ip_geo_timezone": str,
            "order_memo": str,
            "same_card_operations": float64,
            "same_card_operations_day": float64,
            "same_card_operations_fraud": float64,
            "same_card_shared_ip": float64,
            "same_card_shared_ip_day": float64,
            "same_card_shared_ip_fraud": float64,
            "same_card_shared_user": float64,
            "same_card_shared_user_day": float64,
            "same_device_fingerprint": float64,
            "same_device_fingerprint_fraud": float64,
            "same_ip_operations": float64,
            "same_ip_operations_day": float64,
            "same_ip_operations_fraud": float64,
            "same_user_operations_day": float64,
            "tpp_error_reason": str,
            "tpv_alert_indicator": str,
            "tpv_card3d_s2_partition": str,
            "tpv_card_aut3_d_secure2_method": str,
            "tpv_card_dcc": str,
            "transaction_currency": str,
            "transaction_type": str,
            "user_account_status": str,
            "user_agent_browser_family": str,
            "user_agent_device_family": str,
            "user_agent_os_family": str,
            "user_country": str,
            "user_zip": str,
            "phone_country": str,
            "user_category": str
        }

    def getOutlierFields(self): 
        return [
            { "name": 'transaction_amount', "labels": ['small', 'medium', 'big'] },
            { "name": 'user_balance', "labels": ['small', 'medium', 'big'] }
        ]
    
    def getProperties(self):
        # it should be changed to select key from getTypeFields
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
