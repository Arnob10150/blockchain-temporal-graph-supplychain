// Deploy DataProvenance on Hardhat's in-process EVM and measure REAL gas.
const hre = require("hardhat");
const fs = require("fs");
const crypto = require("crypto");

async function main() {
  const C = await hre.ethers.getContractFactory("DataProvenance");
  const c = await C.deploy();
  await c.waitForDeployment();
  const deployTx = c.deploymentTransaction();
  const deployRc = await deployTx.wait();

  const N = 200;
  let regGas = 0n, verGas = 0n;
  const dates = [];
  for (let i = 0; i < N; i++) {
    const date = `2023-${String(1 + (i % 12)).padStart(2,"0")}-${String(1 + (i % 28)).padStart(2,"0")}-${i}`;
    const h = "0x" + crypto.createHash("sha256").update(date + i).digest("hex");
    dates.push([date, h]);
    const tx = await c.registerSnapshot(date, h);
    const rc = await tx.wait();
    regGas += rc.gasUsed;
  }
  // verify is a view function: estimate its gas via provider
  for (let i = 0; i < N; i++) {
    const g = await c.verifySnapshot.estimateGas(dates[i][0], dates[i][1]);
    verGas += g;
  }
  const out = {
    solc: "0.8.19",
    deploy_gas: Number(deployRc.gasUsed),
    register_gas_mean: Number(regGas / BigInt(N)),
    verify_gas_mean: Number(verGas / BigInt(N)),
    registrations: N,
    network: "hardhat in-process EVM"
  };
  fs.writeFileSync("../results/tables/onchain_gas.json", JSON.stringify(out, null, 2));
  console.log(out);
}
main().catch((e) => { console.error(e); process.exit(1); });
