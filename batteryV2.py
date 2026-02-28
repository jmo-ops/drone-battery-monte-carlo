## @file drone_battery_monte_carlo.py
#  @brief Monte Carlo simulation + GUI for comparing drone battery configurations.
#
#  This script runs a Monte Carlo process generator for two battery configurations
#  (e.g., Li-ion vs LiPo) under uncertain wind, temperature, payload, and distance.
#  It reports success probability given a safety reserve requirement and plots
#  distributions of remaining energy.
#
#  Key concepts:
#    - "Nominal" sampling uses uniform distributions across realistic ranges.
#    - "Extreme" sampling uses triangular distributions biased toward worst-case
#      conditions (higher wind, colder temps, heavier payload, longer distance).
#
#  Units:
#    - distance: miles
#    - speed: mph
#    - temperature: °F
#    - wind: mph
#    - payload: lb
#    - energy: Wh
#    - power: W
#
#  @author TEAM 3
#  @date 2026-02-28

import random
import math
import statistics
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# -----------------------------
# Model & Process Generator
# -----------------------------

class BatteryConfig:
    """
    Battery configuration.

    @var capacity_wh Nominal energy capacity (Wh)
    @var base_power_w Baseline power draw (W)
    @var payload_penalty_w_per_lb Added watts per lb payload
    @var wind_penalty_w_per_mph Added watts per mph headwind
    @var cold_capacity_penalty_per_deg Capacity loss fraction per °F below 70
    @var cold_power_penalty_per_deg Power increase fraction per °F below 70
    """
    def __init__(self,
                 key: str,
                 label: str,
                 capacity_wh: float,
                 base_power_w: float,
                 payload_penalty_w_per_lb: float,
                 wind_penalty_w_per_mph: float,
                 cold_capacity_penalty_per_deg: float,
                 cold_power_penalty_per_deg: float = 0.0):
        self.key = key
        self.label = label
        self.capacity_wh = capacity_wh
        self.base_power_w = base_power_w
        self.payload_penalty_w_per_lb = payload_penalty_w_per_lb
        self.wind_penalty_w_per_mph = wind_penalty_w_per_mph
        self.cold_capacity_penalty_per_deg = cold_capacity_penalty_per_deg
        self.cold_power_penalty_per_deg = cold_power_penalty_per_deg


class MissionConfig:
    """
    @brief Mission-level constants used by the simulator.

    The actual mission distance per trial is:
        distance_mi = nominal_distance_mi * distance_factor

    @var nominal_distance_mi Baseline mission distance in miles.
    @var cruise_speed_mph Cruise speed (mph), assumed constant.
    @var safety_energy_fraction Fraction of capacity reserved as safety margin (0–1).
    """
    def __init__(self,
                 nominal_distance_mi: float = 15.0,
                 cruise_speed_mph: float = 25.0,
                 safety_energy_fraction: float = 0.10):
        self.nominal_distance_mi = nominal_distance_mi
        self.cruise_speed_mph = cruise_speed_mph
        self.safety_energy_fraction = safety_energy_fraction


def sample_random_inputs():
    """
    @brief Sample one set of nominal environmental/mission inputs.

    Nominal sampling uses uniform distributions:
    - wind: 0–15 mph
    - temperature: 32–95 °F
    - payload: 0–5 lb
    - distance_factor: 0.90–1.10

    @return (wind_mph, temp_f, payload_lb, distance_factor)
    """
    wind_mph = random.uniform(0.0, 15.0)
    temp_f = random.uniform(32.0, 95.0)
    payload_lb = random.uniform(0.0, 5.0)
    distance_factor = random.uniform(0.9, 1.1)
    return wind_mph, temp_f, payload_lb, distance_factor


def sample_extreme_inputs():
    """
    @brief Sample one set of extreme / worst-case-biased inputs.

    Uses triangular distributions biased toward harsher conditions:
      - wind: 10–20 mph, skewed high
      - temperature: 10–95 °F, skewed low (cold-biased)
      - payload: 2.5–5 lb, skewed high
      - distance_factor: 1.0–1.1, skewed high

    @return (wind_mph, temp_f, payload_lb, distance_factor)
    """
    wind_mph = random.triangular(10.0, 20.0, 20.0)
    temp_f = random.triangular(10.0, 95.0, 10.0)
    payload_lb = random.triangular(2.5, 5.0, 5.0)
    distance_factor = random.triangular(1.0, 1.1, 1.1)
    return wind_mph, temp_f, payload_lb, distance_factor


def effective_capacity_wh(battery: BatteryConfig, temp_f: float) -> float:
    """
    @brief Compute effective capacity (Wh) as a function of temperature.

    Effective capacity decreases below 70°F (linear), clamped to max 50% loss.

    @param battery Battery configuration
    @param temp_f Temperature in °F
    @return Effective capacity in Wh
    """
    if temp_f >= 70.0:
        return battery.capacity_wh

    delta = 70.0 - temp_f
    penalty = battery.cold_capacity_penalty_per_deg * delta
    penalty = min(max(penalty, 0.0), 0.5)
    return battery.capacity_wh * (1.0 - penalty)


def simulate_mission(battery: BatteryConfig, mission: MissionConfig, rng_inputs=None):
    """
    @brief Simulate one mission trial for a given battery under sampled conditions.

    Steps:
      1) sample inputs (wind/temp/payload/distance_factor) unless provided
      2) compute distance_mi = mission.nominal_distance_mi * distance_factor
      3) compute power draw (W) = base + payload penalty + wind penalty
      4) apply cold power penalty if temp < 70°F
      5) compute time = distance / cruise_speed
      6) compute energy_required = power * time
      7) compute effective capacity (cold reduced) and reserve requirement
      8) success if energy_required <= capacity_after_reserve

    @param battery Battery configuration
    @param mission Mission configuration
    @param rng_inputs Optional tuple (wind_mph, temp_f, payload_lb, distance_factor)
    @return dict containing inputs and outputs (distance, remaining energy, success, etc.)
    """
    if rng_inputs is None:
        wind_mph, temp_f, payload_lb, distance_factor = sample_random_inputs()
    else:
        wind_mph, temp_f, payload_lb, distance_factor = rng_inputs

    distance_mi = mission.nominal_distance_mi * distance_factor

    power_w = (
        battery.base_power_w
        + battery.payload_penalty_w_per_lb * payload_lb
        + battery.wind_penalty_w_per_mph * wind_mph
    )
    power_w = max(power_w, 1.0)

    if temp_f < 70.0:
        power_w *= (1.0 + battery.cold_power_penalty_per_deg * (70.0 - temp_f))

    time_h = distance_mi / mission.cruise_speed_mph
    energy_required_wh = power_w * time_h

    cap_eff_wh = effective_capacity_wh(battery, temp_f)
    safety_threshold_wh = cap_eff_wh * mission.safety_energy_fraction
    energy_available_for_mission_wh = cap_eff_wh - safety_threshold_wh

    success = energy_required_wh <= energy_available_for_mission_wh

    energy_remaining_wh = max(cap_eff_wh - energy_required_wh, 0.0)
    energy_remaining_pct = 100.0 * energy_remaining_wh / cap_eff_wh

    return {
        "battery": battery.label,
        "wind_mph": wind_mph,
        "temp_f": temp_f,
        "payload_lb": payload_lb,
        "distance_mi": distance_mi,
        "energy_remaining_pct": energy_remaining_pct,
        "success": success,
    }


def process_generator(num_runs: int, batteries, mission: MissionConfig):
    """@brief Yield one simulation record at a time (nominal inputs)."""
    for _ in range(num_runs):
        for battery in batteries:
            yield simulate_mission(battery, mission)


def process_generator_extreme(num_runs: int, batteries, mission: MissionConfig):
    """@brief Yield one simulation record at a time (extreme-biased inputs)."""
    for _ in range(num_runs):
        for battery in batteries:
            rng_inputs = sample_extreme_inputs()
            yield simulate_mission(battery, mission, rng_inputs=rng_inputs)


def run_monte_carlo(num_runs: int, batteries, mission: MissionConfig):
    """@brief Run nominal Monte Carlo and collect results per battery label."""
    results = {b.label: [] for b in batteries}
    for record in process_generator(num_runs, batteries, mission):
        results[record["battery"]].append(record)
    return results


def run_monte_carlo_extreme(num_runs: int, batteries, mission: MissionConfig):
    """@brief Run extreme-biased Monte Carlo and collect results per battery label."""
    results = {b.label: [] for b in batteries}
    for record in process_generator_extreme(num_runs, batteries, mission):
        results[record["battery"]].append(record)
    return results


def summarize_results(records, battery_config: BatteryConfig, mission_config: MissionConfig):
    """
    @brief Compute summary distribution statistics for one battery’s records.

    @return dict with mean/p5/p95 remaining %, success rate, averages, and deliveries estimate.
    """
    if not records:
        return None

    remaining = [r["energy_remaining_pct"] for r in records]
    successes = [1 if r["success"] else 0 for r in records]
    miles = [r["distance_mi"] for r in records]
    payloads = [r["payload_lb"] for r in records]
    successful_records = [r for r in records if r["success"]]

    def percentile(data, p):
        data_sorted = sorted(data)
        n_local = len(data_sorted)
        idx = (p / 100.0) * (n_local - 1)
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return data_sorted[int(idx)]
        frac = idx - lower
        return data_sorted[lower] * (1 - frac) + data_sorted[upper] * frac

    # Energy usage per mission
    wh_used_list = []
    for r in records:
        cap_eff = effective_capacity_wh(battery_config, r["temp_f"])
        remaining_wh = (r["energy_remaining_pct"] / 100.0) * cap_eff
        wh_used_list.append(cap_eff - remaining_wh)

    avg_wh_per_trip = statistics.mean(wh_used_list) if wh_used_list else 0.0
    usable_wh = battery_config.capacity_wh * (1 - mission_config.safety_energy_fraction)
    missions_per_charge = (usable_wh / avg_wh_per_trip) if avg_wh_per_trip > 0 else 0.0
    deliveries_per_charge = math.floor(missions_per_charge)

    n = len(records)
    mean_remaining = statistics.mean(remaining) if remaining else 0.0
    p5_remaining = percentile(remaining, 5) if remaining else 0.0
    p95_remaining = percentile(remaining, 95) if remaining else 0.0
    success_rate = (sum(successes) / n * 100.0) if n > 0 else 0.0

    avg_miles_all = statistics.mean(miles) if miles else 0.0
    avg_payload_all = statistics.mean(payloads) if payloads else 0.0

    if successful_records:
        avg_miles_success = statistics.mean([r["distance_mi"] for r in successful_records])
        avg_payload_success = statistics.mean([r["payload_lb"] for r in successful_records])
    else:
        avg_miles_success = 0.0
        avg_payload_success = 0.0

    return {
        "mean_remaining": mean_remaining,
        "p5_remaining": p5_remaining,
        "p95_remaining": p95_remaining,
        "success_rate": success_rate,
        "avg_miles_all": avg_miles_all,
        "avg_payload_all": avg_payload_all,
        "avg_miles_success": avg_miles_success,
        "avg_payload_success": avg_payload_success,
        "total_runs": n,
        "successful_runs": sum(successes),
        "avg_wh_per_trip": avg_wh_per_trip,
        "deliveries_per_charge": deliveries_per_charge,
        "missions_per_charge": missions_per_charge,
    }


# -----------------------------
# GUI
# -----------------------------

class MonteCarloGUI:
    """
    @brief Tkinter GUI for running nominal/extreme Monte Carlo simulations.

    Provides:
      - "Simulation" tab: nominal Monte Carlo + histogram plot
      - "Extreme Conditions" tab: extreme-biased Monte Carlo + histogram plot
      - "Parameters Summary" tab: displays configured ranges + battery parameters
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Battery Monte Carlo Process Generator")

        self.battery_a = BatteryConfig(
            key="A",
            label="Li-ion (NMC, cylindrical)",
            capacity_wh=300.0,
            base_power_w=180.0,
            payload_penalty_w_per_lb=10.0,
            wind_penalty_w_per_mph=1.5,
            cold_capacity_penalty_per_deg=0.005,
            cold_power_penalty_per_deg=0.002,
        )
        self.battery_b = BatteryConfig(
            key="B",
            label="LiPo (pouch, high-power)",
            capacity_wh=260.0,
            base_power_w=170.0,
            payload_penalty_w_per_lb=11.0,
            wind_penalty_w_per_mph=1.2,
            cold_capacity_penalty_per_deg=0.004,
            cold_power_penalty_per_deg=0.0015,
        )
        self.mission = MissionConfig(
            nominal_distance_mi=15.0,
            cruise_speed_mph=25.0,
            safety_energy_fraction=0.10,
        )

        self.batteries = [self.battery_a, self.battery_b]

        self.figure = None
        self.canvas = None
        self.extreme_figure = None
        self.extreme_canvas = None

        self._build_tabs()
 
    ## @brief Constructs the tabbed interface for the application.
    #  @details Uses a ttk.Notebook to create three distinct sections:
    #  1. Simulation: For standard Monte Carlo runs.
    #  2. Parameters Summary: Displays a read-only text summary of battery profiles.
    #  3. Extreme Conditions: For high-stress "torture tests" with biased inputs
    def _build_tabs(self):
        notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "Custom.TNotebook.Tab",
            background="#4A6FFF",
            foreground="white",
            padding=[10, 5],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", "#2E47B8")],
            foreground=[("selected", "white")],
        )

        notebook.pack(fill="both", expand=True)

        sim_tab = ttk.Frame(notebook)
        notebook.add(sim_tab, text="Simulation")
        self._build_widgets(sim_tab)

        params_tab = ttk.Frame(notebook)
        notebook.add(params_tab, text="Parameters Summary")

        extreme_tab = ttk.Frame(notebook)
        notebook.add(extreme_tab, text="Extreme Conditions")
        self._build_extreme_widgets(extreme_tab)

        params_text = tk.Text(params_tab, wrap="word")
        params_text.pack(fill="both", expand=True, padx=10, pady=10)
        params_text.insert(tk.END, self._generate_parameters_summary())
        params_text.configure(state="disabled")
 
    ## @brief Builds the primary simulation dashboard.
    #  @details Sets up the 'Simulation' tab, which includes:
    #  - A 'Simulation Settings' group for configuring run counts.
    #  - A 'Summary (Distribution)' group for displaying output statistics.
    #  @param parent The parent widget (typically the simulation tab frame).
    def _build_widgets(self, parent):
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main_frame, text="Simulation Settings")
        controls.pack(side="top", fill="x", pady=5)

        ttk.Label(controls, text="Number of runs per battery:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.num_runs_var = tk.StringVar(value="500")
        ttk.Entry(controls, textvariable=self.num_runs_var, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(controls, text="Run Simulation", command=self.run_simulation).grid(row=0, column=2, sticky="w", padx=10, pady=5)

        summary_frame = ttk.LabelFrame(main_frame, text="Summary (Distribution)")
        summary_frame.pack(side="top", fill="both", expand=True, pady=5)

        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill="both", expand=True, padx=5, pady=5)

        self.summary_text = tk.Text(summary_inner, height=20, wrap="word")
        self.summary_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(summary_inner, orient="vertical", command=self.summary_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.summary_text.configure(yscrollcommand=scrollbar.set)

        plot_frame = ttk.LabelFrame(main_frame, text="Distributions (Remaining Energy)")
        plot_frame.pack(side="top", fill="both", expand=True, pady=5)
        self.plot_container = plot_frame

    def _build_extreme_widgets(self, parent):
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main_frame, text="Extreme Simulation Settings")
        controls.pack(side="top", fill="x", pady=5)

        ttk.Label(controls, text="Number of runs per battery:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.extreme_runs_var = tk.StringVar(value="500")
        ttk.Entry(controls, textvariable=self.extreme_runs_var, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(controls, text="Extreme sampling biases wind high, temp low, payload high, distance high.").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2
        )

        ttk.Button(controls, text="Run Extreme Simulation", command=self.run_extreme_simulation).grid(
            row=0, column=2, sticky="w", padx=10, pady=5
        )

        summary_frame = ttk.LabelFrame(main_frame, text="Extreme Summary (Distribution)")
        summary_frame.pack(side="top", fill="both", expand=True, pady=5)

        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill="both", expand=True, padx=5, pady=5)

        self.extreme_summary_text = tk.Text(summary_inner, height=20, wrap="word")
        self.extreme_summary_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(summary_inner, orient="vertical", command=self.extreme_summary_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.extreme_summary_text.configure(yscrollcommand=scrollbar.set)

        plot_frame = ttk.LabelFrame(main_frame, text="Extreme Distributions (Remaining Energy)")
        plot_frame.pack(side="top", fill="both", expand=True, pady=5)
        self.extreme_plot_container = plot_frame

    ## @brief Executes the standard Monte Carlo simulation.
    #  @details Performs the following process:
    #  1. Validates user input for the number of runs.
    #  2. Calls run_monte_carlo() to perform stochastic trials for each battery.
    #  3. Generates statistical summaries (mean, P5, P95) for mission results.
    #  4. Calculates energy usage metrics and delivery throughput.
    #  5. Performs a delta comparison if multiple batteries are configured.
    #  @exception ValueError Raised if the run count is not a positive integer.

    def run_simulation(self):
        try:
            num_runs = int(self.num_runs_var.get())
            if num_runs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of runs must be a positive integer.")
            return

        results = run_monte_carlo(num_runs, self.batteries, self.mission)

        summaries = {}
        self.summary_text.delete("1.0", tk.END)

        for battery in self.batteries:
            recs = results.get(battery.label, [])
            summary = summarize_results(recs, battery_config=battery, mission_config=self.mission)
            if not summary:
                continue

            summaries[battery.label] = summary

            self.summary_text.insert(
                tk.END,
                f"{battery.label} — Monte Carlo Results\n"
                f"  Total simulation trials: {summary['total_runs']}\n"
                f"  Successful mission trials (met reserve requirement): "
                f"{summary['successful_runs']} / {summary['total_runs']} "
                f"({summary['success_rate']:.1f}% success probability)\n\n"
                f"  Remaining Energy Distribution:\n"
                f"    Mean remaining energy: {summary['mean_remaining']:.1f}%\n"
                f"    5th percentile remaining: {summary['p5_remaining']:.1f}%\n"
                f"    95th percentile remaining: {summary['p95_remaining']:.1f}%\n\n"
                f"  Operational Averages:\n"
                f"    Avg miles (all trials): {summary['avg_miles_all']:.2f} mi\n"
                f"    Avg payload (all trials): {summary['avg_payload_all']:.2f} lb\n"
                f"    Avg miles (successful only): {summary['avg_miles_success']:.2f} mi\n"
                f"    Avg payload (successful only): {summary['avg_payload_success']:.2f} lb\n\n"
                f"  Energy Use & Deliveries:\n"
                f"    Avg energy used per mission: {summary['avg_wh_per_trip']:.1f} Wh\n"
                f"    Missions per charge (float): {summary['missions_per_charge']:.2f}\n"
                f"    Approx. deliveries per full charge (floor): {summary['deliveries_per_charge']}\n\n"
            )

        if len(self.batteries) >= 2:
            a_name = self.batteries[0].label
            b_name = self.batteries[1].label
            if a_name in summaries and b_name in summaries:
                A = summaries[a_name]
                B = summaries[b_name]
                self.summary_text.insert(
                    tk.END,
                    "=============================\n"
                    "Li-ion vs LiPo  Comparison (Nominal)\n"
                    "=============================\n"
                    f"Δ Mean remaining (Li-ion - LiPo): {A['mean_remaining'] - B['mean_remaining']:.2f} %\n"
                    f"Δ 5th pct remaining (Li-ion - LiPo): {A['p5_remaining'] - B['p5_remaining']:.2f} %\n"
                    f"Δ 95th pct remaining (Li-ion - LiPo): {A['p95_remaining'] - B['p95_remaining']:.2f} %\n"
                    f"Δ Avg Wh/mission (Li-ion - LiPo): {A['avg_wh_per_trip'] - B['avg_wh_per_trip']:.2f} Wh\n"
                    f"Δ Missions/charge (Li-ion - LiPo): {A['missions_per_charge'] - B['missions_per_charge']:.2f}\n"
                    f"Success rate A: {A['success_rate']:.1f}%   |   Success rate B: {B['success_rate']:.1f}%\n\n"
                )

        self._plot_distributions(results)

    ## @brief Constructs the "Extreme Conditions" dashboard.
    #  @details Sets up a high-stress testing environment including:
    #  - A 'Extreme Simulation Settings' frame for run count configuration.
    #  - An informational label describing the "Extreme" stochastic biases (wind, temp, etc.).
    #  - A scrollable summary text area for distribution results.
    #  - A plot container for visualizing energy distributions.
    #  @param parent The parent frame (typically the Extreme Conditions tab).
    def run_extreme_simulation(self):
        try:
            num_runs = int(self.extreme_runs_var.get())
            if num_runs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of runs must be a positive integer.")
            return

        results = run_monte_carlo_extreme(num_runs, self.batteries, self.mission)

        summaries = {}
        self.extreme_summary_text.delete("1.0", tk.END)

        for battery in self.batteries:
            recs = results.get(battery.label, [])
            summary = summarize_results(recs, battery_config=battery, mission_config=self.mission)
            if not summary:
                continue

            summaries[battery.label] = summary

            self.extreme_summary_text.insert(
                tk.END,
                f"{battery.label} — EXTREME Monte Carlo Results\n"
                f"  Total simulation trials: {summary['total_runs']}\n"
                f"  Successful mission trials (met reserve requirement): "
                f"{summary['successful_runs']} / {summary['total_runs']} "
                f"({summary['success_rate']:.1f}% success probability)\n\n"
                f"  Remaining Energy Distribution:\n"
                f"    Mean remaining energy: {summary['mean_remaining']:.1f}%\n"
                f"    5th percentile remaining: {summary['p5_remaining']:.1f}%\n"
                f"    95th percentile remaining: {summary['p95_remaining']:.1f}%\n\n"
                f"  Operational Averages:\n"
                f"    Avg miles (all trials): {summary['avg_miles_all']:.2f} mi\n"
                f"    Avg payload (all trials): {summary['avg_payload_all']:.2f} lb\n"
                f"    Avg miles (successful only): {summary['avg_miles_success']:.2f} mi\n"
                f"    Avg payload (successful only): {summary['avg_payload_success']:.2f} lb\n\n"
                f"  Energy Use & Deliveries:\n"
                f"    Avg energy used per mission: {summary['avg_wh_per_trip']:.1f} Wh\n"
                f"    Missions per charge (float): {summary['missions_per_charge']:.2f}\n"
                f"    Approx. deliveries per full charge (floor): {summary['deliveries_per_charge']}\n\n"
                "  Extreme input bias:\n"
                "    Wind: high-biased (10–20 mph)\n"
                "    Temp: low-biased (10–95 °F)\n"
                "    Payload: high-biased (2.5–5 lb)\n"
                "    Distance factor: high-biased (1.0–1.1)\n\n"
            )

        if len(self.batteries) >= 2:
            a_name = self.batteries[0].label
            b_name = self.batteries[1].label
            if a_name in summaries and b_name in summaries:
                A = summaries[a_name]
                B = summaries[b_name]
                self.extreme_summary_text.insert(
                    tk.END,
                    "=============================\n"
                    "Li-ion vs LiPo  Comparison (Extreme)\n"
                    "=============================\n"
                    f"Δ Mean remaining (Li-ion - LiPo): {A['mean_remaining'] - B['mean_remaining']:.2f} %\n"
                    f"Δ 5th pct remaining (Li-ion - LiPo): {A['p5_remaining'] - B['p5_remaining']:.2f} %\n"
                    f"Δ 95th pct remaining (Li-ion - LiPo): {A['p95_remaining'] - B['p95_remaining']:.2f} %\n"
                    f"Δ Avg Wh/mission (Li-ion - LiPo): {A['avg_wh_per_trip'] - B['avg_wh_per_trip']:.2f} Wh\n"
                    f"Δ Missions/charge (Li-ion - LiPo): {A['missions_per_charge'] - B['missions_per_charge']:.2f}\n"
                    f"Success rate A: {A['success_rate']:.1f}%   |   Success rate B: {B['success_rate']:.1f}%\n\n"
                )

        self._plot_extreme_distributions(results)
  
    ## @brief Updates the histogram display for extreme simulation results.
    #  @details Functionally identical to _plot_distributions but targets the 
    #  Extreme Conditions UI container and uses biased result data.
    #  @param results A dictionary containing extreme/biased simulation data.
    def _plot_distributions(self, results):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        self.figure = Figure(figsize=(6, 4))
        ax = self.figure.add_subplot(111)

        for battery in self.batteries:
            remaining = [r["energy_remaining_pct"] for r in results[battery.label]]
            ax.hist(remaining, bins=20, alpha=0.5, label=battery.label, edgecolor="black")

        ax.set_xlabel("Remaining Energy (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo Distributions of Remaining Energy")
        ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    ## @brief Refreshes the histogram for extreme simulation results.
    def _plot_extreme_distributions(self, results):
        if self.extreme_canvas is not None:
            self.extreme_canvas.get_tk_widget().destroy()
            self.extreme_canvas = None

        self.extreme_figure = Figure(figsize=(6, 4))
        ax = self.extreme_figure.add_subplot(111)

        for battery in self.batteries:
            remaining = [r["energy_remaining_pct"] for r in results[battery.label]]
            ax.hist(remaining, bins=20, alpha=0.5, label=battery.label, edgecolor="black")

        ax.set_xlabel("Remaining Energy (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("EXTREME Monte Carlo Distributions of Remaining Energy")
        ax.legend()

        self.extreme_canvas = FigureCanvasTkAgg(self.extreme_figure, master=self.extreme_plot_container)
        self.extreme_canvas.draw()
        self.extreme_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    ## @brief Generates a formatted string summarizing all mission and battery constants.
    #  @details Aggregates nominal/extreme sampling ranges, mission setup (distance, speed), 
    #  and detailed battery profiles (capacity, penalties) for the UI summary tab.
    #  @return A multi-line string containing the full parameter report.
    def _generate_parameters_summary(self):
        return (
            "Environmental Ranges (per experiment)\n"
            "-------------------------------------\n"
            "Nominal sampling:\n"
            "  Wind:            0–15 mph\n"
            "  Temperature:     32–95 °F\n"
            "  Payload:         0–5 lb\n"
            "  Distance factor: 0.90–1.10 (±10%)\n\n"
            "Extreme sampling (biased):\n"
            "  Wind:            10–20 mph (high-biased)\n"
            "  Temperature:     10–95 °F (cold-biased)\n"
            "  Payload:         2.5–5 lb (high-biased)\n"
            "  Distance factor: 1.0–1.1 (high-biased)\n\n"
            "Mission Setup\n"
            "-------------\n"
            f"Nominal distance:  {self.mission.nominal_distance_mi} miles\n"
            f"Cruise speed:      {self.mission.cruise_speed_mph} mph\n"
            f"Safety reserve:    {self.mission.safety_energy_fraction * 100:.0f}%\n\n"
            "Battery Profiles\n"
            "----------------\n"
            f"{self.battery_a.label}:\n"
            f"  Capacity:        {self.battery_a.capacity_wh} Wh\n"
            f"  Base power:      {self.battery_a.base_power_w} W\n"
            f"  Payload penalty: {self.battery_a.payload_penalty_w_per_lb} W/lb\n"
            f"  Wind penalty:    {self.battery_a.wind_penalty_w_per_mph} W/mph\n"
            f"  Cold capacity:   {self.battery_a.cold_capacity_penalty_per_deg} per °F < 70\n"
            f"  Cold power:      {self.battery_a.cold_power_penalty_per_deg} per °F < 70\n\n"
            f"{self.battery_b.label}:\n"
            f"  Capacity:        {self.battery_b.capacity_wh} Wh\n"
            f"  Base power:      {self.battery_b.base_power_w} W\n"
            f"  Payload penalty: {self.battery_b.payload_penalty_w_per_lb} W/lb\n"
            f"  Wind penalty:    {self.battery_b.wind_penalty_w_per_mph} W/mph\n"
            f"  Cold capacity:   {self.battery_b.cold_capacity_penalty_per_deg} per °F < 70\n"
            f"  Cold power:      {self.battery_b.cold_power_penalty_per_deg} per °F < 70\n\n"
            "Note:\n"
            "  Chemistry differences are represented by parameter differences\n"
            "  (capacity, power penalties, and cold sensitivity).\n"
        )
       
def generate_docs():
   """! Executes the Doxygen command to refresh documentation."""
   try:
       subprocess.run(["doxygen", "Doxyfile"], check=True)
       print("Documentation generated successfully.")
   except FileNotFoundError:
       print("Error: Doxygen not found in system path.")

## @brief Main entry point for the application.
def main():
    root = tk.Tk()
    app = MonteCarloGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    generate_docs()