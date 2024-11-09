from flask import Flask, jsonify

from jass.game.game_observation import GameObservation
from requests import request

from jass.game.game_util import *

def http_play_card(play_card_int):
    play_card_str = convert_int_encoded_cards_to_str_encoded(play_card_int)


from flask import Flask, jsonify

app = Flask(__name__)



@app.route('/select_trump', methods=['POST'])
def select_trump():
    obs = GameObservation.from_json(request.get_json())
    # get_trump
    """
        0 = diamonds
        1 = hearts
        2 = spades
        3 = clubs
        4 = obeabe
        5 = uneufe
        10 = push
    """
    trump = 3
    response = {
        "trump": trump
    }
    return jsonify(response), 200

@app.route('/play_card', methods=['POST'])
def play_card():
    obs = GameObservation.from_json(request.get_json())

    # get move
    move = "C8"
    response = {
        "card": move
    }
    return jsonify(response), 200



if __name__ == '__main__':
    app.run(debug=True)
