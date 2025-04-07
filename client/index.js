const serverConnectInput = document.getElementById("server-connection-params");
const serverConnectButton = document.getElementById("server-connection-button");
const serverConnectStatusLabel = document.getElementById(
  "server-connection-status"
);

let serverBaseURL = "";

/**
 * Beat the URL a little bit so it can be used for uh, connections.
 * @param {String} input - The string to beat into shape.
 */
const formatServerURL = (input) => {
  let processedUrl = input;
  if (!processedUrl.endsWith("/")) {
    processedUrl += "/";
  }
  // This ... doesn't work?
  //   if (!processedUrl.startsWith("http://")) {
  //     processedUrl = `http://${processedUrl}`;
  //     console.log(processedUrl);
  //   }
  return processedUrl;
};

/**
 * Check the health of the sde server. Return true with a null rationale if we're
 * successful, and false with an error otherwise.
 * @param {String} url -- The URL where we'll check the health of the sde server
 */
const healthCheckServerRequest = async (url) => {
  try {
    const healthCheckResult = await fetch(url);
    if (!healthCheckResult.ok) {
      return [false, `Received ${healthCheckResult.status} fom "${url}".`];
    }
    return [true, null];
  } catch (error) {
    return [false, `Error while asking for status at "${url}": ${error}`];
  }
};

const addConnectionEventListeners = async () => {
  serverConnectButton.addEventListener("click", () => {
    const inputServerURL = formatServerURL(serverConnectInput.value);
    if (inputServerURL === "") {
      serverConnectStatusLabel.innerText = "Enter a value.";
      return;
    }
    healthCheckServerRequest(inputServerURL).then(([isOK, message]) => {
      if (isOK === false) {
        serverConnectStatusLabel.innerText = message;
      } else {
        serverConnectStatusLabel.style.color = "green";
        serverConnectStatusLabel.innerText = "Connected!";
        serverBaseURL = inputServerURL;
        setInterval(() => {
          healthCheckServerRequest(serverConnectInput.value).then(
            ([recheckOK, recheckMessage]) => {
              const recheckURL = formatServerURL(serverConnectInput.value);
              if (recheckOK === false) {
                serverConnectStatusLabel.style.color = "red";
                serverConnectStatusLabel.innerText = `Connection to "${recheckURL}" lost!`;
              } else if (recheckOK === true) {
                serverConnectStatusLabel.style.color = "green";
                serverConnectStatusLabel.innerText = "Connected!";
                serverBaseURL = recheckURL;
              }
            }
          );
        }, 3000);
      }
    });
  });
};

const main = () => {
  addConnectionEventListeners();
};

main();
