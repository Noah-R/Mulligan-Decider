//this is an unused file, it contains incomplete code for predicting client-side, i'm trying to do all predicting server-side, but leaving this here just in case
//model.json is converted from model_30_sep_2021_2
//add this tag to html: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
async function load() {
    const model = await tf.loadLayersModel('model.json');
    return model;
};
const model = load();
function predict(model) {
    const listOfTensors = {}
    listOfTensors["cards"]=(tf.tensor([parseFloat(hand.length)/7.0], name="cards"));
    listOfTensors["on_play"]=(tf.tensor([parseFloat(document.getElementById('onplay').checked)], name="on_play"));
    for(var x=0; x<cards.length; x++){
        listOfTensors[cards[x]]=[0.0]
    }
    for(var x=0; x<hand.length; x++){
        listOfTensors[hand[x]][0]+=1.0/7.0
    }
    for(var x=0; x<cards.length; x++){
        listOfTensors[cards[x]]=(tf.Tensor(listOfTensors[cards[x]], name=cardToColumn(cards[x])))
    }

    model.then(model => {
        let result = model.predict(Object.values(listOfTensors));
        alert(result);
        result = result.round().dataSync()[0];
        alert(result);
    });
};
function cardToColumn(cardname){
    return "opening_hand_"+cardname.replace(" ", "_").replace(",", "").replace("'", "")
}