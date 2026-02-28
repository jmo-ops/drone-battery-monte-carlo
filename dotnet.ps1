@"
digraph G {
  rankdir=LR;
  node [shape=box];
  "BatteryConfig" -> "simulate_mission";
  "MissionConfig" -> "simulate_mission";
  "simulate_mission" -> "summarize_results";
}
"@ | Set-Content -Encoding UTF8 diagram.dot