# -*- coding:utf8 -*-
# File   : configs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/28/17
# 
# This file is part of TensorArtist


NS_CLEANUP_WAIT = 10

NS_CTL_PROTOCAL = 'tcp'
NS_CTL_HOST = '*'
NS_CTL_PORT = '43521'
NS_HEARTBEAT_INTERVAL = 3

CTL_HEARTBEAT_INTERVAL = 5
CTL_CTL_SEND_COUNTDOWN = 5
CTL_DAT_SEND_COUNTDOWN = 5
CTL_DAT_PROTOCAL = 'tcp'
CTL_DAT_HOST = '*'
CTL_DAT_HWM = 5


class Actions:
    NS_REGISTER_CTL = 'ns-register-ctl'
    NS_REGISTER_CTL_SUCC = 'ns-register-ctl-succ'

    NS_REGISTER_OPIPE = 'ns-register-opipe'
    NS_REGISTER_OPIPE_SUCC = 'ns-register-opipe-succ'

    NS_HEARTBEAT = 'ns-heartbeat'
    NS_HEARTBEAT_SUCC = 'ns-heartbeat-succ'

    NS_QUERY_OPIPE = 'ns-query-opipe'
    NS_QUERY_OPIPE_RESPONSE = 'ns-query-opipe-response'

    NS_CLOSE_CTL = 'ns-close-ctl'
    NS_CLOSE_CTL_SUCC = 'ns-close-ctl-succ'
    NS_OPEN_CTL = 'ns-open-ctl'
    NS_OPEN_CTL_SUCC = 'ns-open-ctl-succ'

    CTL_HEARTBEAT = 'ctl-heartbeat'
    CTL_HEARTBEAT_SUCC = 'ctl-heartbeat-succ'

    CTL_OPEN = 'ctl-open'
    CTL_OPEN_RESPONSE = 'ctl-open-response'
    CTL_OPENED = 'ctl-opened'
    CTL_OPENED_SUCC = 'ctl-opened-succ'
