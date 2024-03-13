import pandas as pd
from typing import Dict, List
import numpy as np
from typing import Union

GENDER = 'gender'
EXHAUST_VOLUME = 'exhaust_volume'
MOTORCYCLE_STYLE = 'motorcycle_style'
PASSENGER_ON_BACKSEAT = 'passenger_on_backseat'
HELMET = 'helmet'
MAN_HEIGHT = 'man_height'
WOMAN_HEIGHT = 'woman_height'
MAN_WEIGHT = 'man_weight'
WOMAN_WEIGHT = 'woman_weight'
AGE = 'age'
PURCHASE_PRICE = 'purchase_price'
MONTHLY_MILEAGE = 'monthly_mileage'
TRAFFIC_TICKET_COUNT = 'traffic_ticket_count'
SPEEDING_HABBIT = 'speeding_habbit'
HAS_LICENSE = 'has_license'
HAS_MODIFICATIONS = 'has_modifications'
IS_NOISY_ENGINE = 'is_noisy_engine'
IS_MOUNTAIN_RIDE = 'is_mountain_ride'
HAS_REGULAR_MAINTENANCE = 'has_regular_maintenance'
COMMUTE_WITH_MOTORCYCLE = 'commute_with_motorcycle'
LABEL = 'class'
HIGH_RISK = 'high_risk'
MEDIUM_RISK = 'medium_risk'
LOW_RISK = 'low_risk'


ATTRIBUTES = {
    #categorical
    GENDER: ['man', 'woman'],
    EXHAUST_VOLUME: ['50', '100', '125', '150'],
    MOTORCYCLE_STYLE: ['DRG', 'JET', 'cygnus', 'GP', 'duke', 'Many', 'famous', 'Fiddle', 'Cuxi', 'Limi', 'axis Z', 'Swish', 'POG', 'FORCE', 'BWS'],
    PASSENGER_ON_BACKSEAT:['never', 'seldom', 'sometimes', 'usually', 'always'],
    HELMET:['1', '2', '3'],
    # 1'西瓜皮', 2'普通安全帽', 3'全罩式'

    #gussian continuous
    MAN_HEIGHT: [173, 8],
    WOMAN_HEIGHT: [158, 8],
    MAN_WEIGHT: [70, 10],
    WOMAN_WEIGHT: [50, 10],
    AGE: [23, 15],

    #discrete
    PURCHASE_PRICE: list(range(40, 130+1)), #40k-130k
    MONTHLY_MILEAGE: list(range(0, 1000+1)), 
    TRAFFIC_TICKET_COUNT: list(range(0, 10+1)),
    SPEEDING_HABBIT: list(range(0, 30)),

    #boolean
    HAS_LICENSE: [0, 1],
    HAS_MODIFICATIONS: [0, 1], #excpet engine
    IS_NOISY_ENGINE: [0, 1],
    IS_MOUNTAIN_RIDE: [0, 1],
    HAS_REGULAR_MAINTENANCE: [0, 1],
    COMMUTE_WITH_MOTORCYCLE: [0, 1],
}

PROBS = {
    # GENDER:[ 0.5, 0.5], 
    EXHAUST_VOLUME: [0.1, 0.1, 0.6, 0.2],
    # MOTORCYCLE_STYLE: [],
    PASSENGER_ON_BACKSEAT: [0.1, 0.4, 0.25, 0.15, 0.1],
    HELMET: [0.1, 0.6, 0.3],
    HAS_LICENSE: [0.1, 0.9],
    HAS_MODIFICATIONS: [0.7, 0.3],
    IS_NOISY_ENGINE: [0.8, 0.2],
    IS_MOUNTAIN_RIDE: [0.7, 0.3],
    COMMUTE_WITH_MOTORCYCLE: [0.2, 0.8],
}

def gen_data(attributes: Dict[str, List[str]]) -> Dict[str, Union[int, float]]:
    data_row = {}
    # categorical = [GENDER, EXHAUST_VOLUME, MOTORCYCLE_STYLE, PASSENGER_ON_BACKSEAT, HELMET]
    # discrete = [AGE, PURCHASE_PRICE, MONTHLY_MILEAGE, TRAFFIC_TICKET_COUNT, SPEEDING_HABBIT]
    # boolean = [HAS_LICENSE, HAS_MODIFICATIONS, IS_NOISY_ENGINE, IS_MOUNTAIN_RIDE, HAS_REGULAR_MAINTENANCE, COMMUTE_WITH_MOTORCYCLE]
    gaussian = [MAN_HEIGHT, WOMAN_HEIGHT, MAN_WEIGHT, WOMAN_WEIGHT, AGE]
    for attr in attributes:
        if attr in PROBS:
            data_row[attr] = np.random.choice(attributes[attr], p = PROBS[attr])
        elif attr in gaussian:
            mu, sigma = attributes[attr]
            if data_row[GENDER] == 'man':
                if attr == MAN_WEIGHT:
                    data_row['weight'] = round(np.random.normal(mu, sigma), 3)
                elif attr == MAN_HEIGHT:
                    data_row['height'] = round(np.random.normal(mu, sigma), 3)
    
            elif data_row[GENDER] == 'woman':
                if attr == WOMAN_WEIGHT:
                    data_row['weight'] = round(np.random.normal(mu, sigma), 3)
                elif attr == WOMAN_HEIGHT:
                    data_row['height'] = round(np.random.normal(mu, sigma), 3)

            if attr == AGE:
                data_row[attr] = round(np.random.normal(mu, sigma), 3)
                while data_row[attr] < 18:
                    data_row[attr] = round(np.random.normal(mu, sigma), 3)
        else:
            data_row[attr] = np.random.choice(attributes[attr])

    if is_high_risk(data_row):
        data_row[LABEL] = HIGH_RISK
    elif is_medium_risk(data_row):
        data_row[LABEL] = MEDIUM_RISK
    else:
        data_row[LABEL] = LOW_RISK
        
    return data_row

def gen_dataset(N:int):
    data_rows = []
    print('========== Gen Dataset ==========')
    for i in range(N):
        data = gen_data(attributes=ATTRIBUTES)
        data_rows.append(data)
    # messup_count = int(N * messup_ratio)
    # messup_ids = np.random.choice(range(N), messup_count)
    # for xid in messup_ids:
    #     data_rows[xid][LABEL] = np.random.choice(
    #         [x for x in Classes if x != data_rows[xid][LABEL]])
    print(f'Hyperprameters: data count: {N}')
    # print(f'Messup data count: {len(messup_ids)}')

    df = pd.DataFrame(data_rows)
    return df


def is_high_risk(d: Dict[str, Union[int, float]]):
    """
    判斷條件(1): 山道猴子
    (a)必要條件
    man
    age 18~30
    排氣量 125 or 150
    Has_modifications = 1

    (b)滿足所有必要條件且滿足下列其中一個
    noisy engine = 1
    purchase price > 10k
    traffic ticket count >= 5
    MOTORCYCLE_STYLE: DRG JET FORCE CYGNUS BWS 
    """

    if (
        d[GENDER] == 'man' and
        18 <= d[AGE] <= 30 and
        d[EXHAUST_VOLUME] in ['125', '150'] and
        d[HAS_MODIFICATIONS] == 1 and
        d[IS_MOUNTAIN_RIDE] == 1 
    ):
        if (
            d[IS_NOISY_ENGINE] == 1 or
            d[PURCHASE_PRICE] > 100 or
            # d[TRAFFIC_TICKET_COUNT] >= 5 or
            d[MOTORCYCLE_STYLE] in ['DRG', 'JET', 'FORCE', 'CYGNUS', 'BWS']
        ):
            return True

    """
    判斷條件(2): 在地無照老人三寶
    (a)必要條件
    age > 50
    has license = 0 (no license)
    排氣量 50 or 100 or 125
    helmet 1 or 2
    traffic ticket count >= 3
    """

    if (
        d[AGE] > 50 and
        d[HAS_LICENSE] == 0 and
        d[EXHAUST_VOLUME] in ['50', '100', '125'] and
        d[HELMET] in ['1', '2'] #and
        # d[TRAFFIC_TICKET_COUNT] >= 3
    ):
        return True  

    """
    判斷條件(3): 市區飄車仔
    (a)必要條件
    helmet 1
    SPEEDING_HABBIT >= 15
    Has_modifications = 1

    (b)滿足所有必要條件且滿足下列其中一個
    Is_noisy_engine =  1
    traffic ticket count >= 3
    MOTORCYCLE_STYLE: Many CUXI  (南Many 北Cuxi)
    """

    if (
        d[HELMET] == '1' and
        d[SPEEDING_HABBIT] >= 15 and
        d[HAS_MODIFICATIONS] == 1
    ):
        if (
            d[IS_NOISY_ENGINE] == 1 or
            # d[TRAFFIC_TICKET_COUNT] >= 3 or
            d[MOTORCYCLE_STYLE] in ['Many', 'Cuxi']
        ):
            return True

    return False


def is_medium_risk(d: Dict[str, Union[int, float]]):
    
    """
    判斷條件(4): 普通機車通勤族
    (a)必要條件
    排氣量 100 or 125
    helmet 2
    mothly mileage > 600km
    COMMUTE_WITH_MOTORCYCLE = 1
    """

    if (
        d[EXHAUST_VOLUME] in ['100', '125'] and
        d[HELMET] == '2' and
        d[COMMUTE_WITH_MOTORCYCLE] == 1 #and
        # d[TRAFFIC_TICKET_COUNT] <= 2
    ):
        return True

    """
    判斷條件(5): 車子快拋錨騎很慢的人
    (a)必要條件 
    HAS_REGULAR_MAINTENANCE = 0
    SPEEDING_HABBIT = 0
    age > 40
    """

    if (
        d[HAS_REGULAR_MAINTENANCE] == 0 and
        d[SPEEDING_HABBIT] == 0 and
        d[AGE] > 40 and
        d[PURCHASE_PRICE] < 80
    ):
        return True

    return False  


if __name__ == '__main__':
    N = 7000

    dataset1 = gen_dataset(N)
    file_path = f"inputs/dataset1-{N}.csv"
    dataset1.to_csv(file_path, index = False)
    print("Output dataset1")

    N = 3000
    dataset2 = gen_dataset(N)
    file_path = f"inputs/dataset2-{N}.csv"
    dataset2.to_csv(file_path, index = False)
    print("Output dataset2")

    print('==================================')
    # print('Output complete.')