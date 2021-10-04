const express = require('express');
const app = express();
app.use(express.static('public'));

app.get('/', function (req, res) {
  res.send("Nothing at base directory, try /index.html");
});

app.get('/*'), (req, res) => {
    res.send("Something went wrong here, Uniform Golf");
  }

let port = process.env.PORT;
if (port == null || port == "") {
  port = 3000;
}
app.listen(port, () => console.log(`Running at localhost:`+port));

// cd C:\Users\noahr\Desktop\Mulligan-Decider\webapp
// node index.js
// http://localhost:3000/index.html