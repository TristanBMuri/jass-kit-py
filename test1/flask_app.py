from flask import Flask, jsonify, request

from jass.game.game_observation import GameObservation

from jass.game.game_util import *
from jass.agents.agent_rule_based import AgentRuleBased
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

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
            "trump": AgentRuleBased().action_trump(obs)
    }

    return jsonify(response), 200

@app.route('/play_card', methods=['POST'])
def play_card():
    obs = GameObservation.from_json(request.get_json())



    # get move
    response = {
        "card": convert_int_encoded_cards_to_str_encoded([AgentRuleBased().action_play_card(obs)])[0]
    }
    return jsonify(response), 200



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
