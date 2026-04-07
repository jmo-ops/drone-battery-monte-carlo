## @file batteryV2_improved.py
#  @brief Monte Carlo simulation + GUI for comparing drone battery configurations.
#
#  Improvements in this version:
#    - clearer top-line KPI cards
#    - better formatted summary text
#    - two plots per run (remaining energy % and trip energy Wh)
#    - chemistry-aligned assumptions: Li-ion carries more nominal energy than LiPo
#    - updated mission reserve to 20%
#    - includes 50 lb empty-drone baseline in the mission power model

import random
import math
import statistics
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    matplotlib.use("Agg")
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure


# -----------------------------
# Model & Process Generator
# -----------------------------

class BatteryConfig:
    """Battery configuration."""

    def __init__(
        self,
        key: str,
        label: str,
        capacity_wh: float,
        base_power_w: float,
        payload_penalty_w_per_lb: float,
        wind_penalty_w_per_mph: float,
        cold_capacity_penalty_per_deg: float,
        cold_power_penalty_per_deg: float = 0.0,
    ):
        self.key = key
        self.label = label
        self.capacity_wh = capacity_wh
        self.base_power_w = base_power_w
        self.payload_penalty_w_per_lb = payload_penalty_w_per_lb
        self.wind_penalty_w_per_mph = wind_penalty_w_per_mph
        self.cold_capacity_penalty_per_deg = cold_capacity_penalty_per_deg
        self.cold_power_penalty_per_deg = cold_power_penalty_per_deg


class MissionConfig:
    """Mission-level constants used by the simulator."""

    def __init__(
        self,
        nominal_distance_mi: float = 15.0,
        cruise_speed_mph: float = 25.0,
        safety_energy_fraction: float = 0.20,
        empty_drone_weight_lb: float = 50.0,
        weight_penalty_w_per_lb: float = 1.0,
    ):
        self.nominal_distance_mi = nominal_distance_mi
        self.cruise_speed_mph = cruise_speed_mph
        self.safety_energy_fraction = safety_energy_fraction
        self.empty_drone_weight_lb = empty_drone_weight_lb
        self.weight_penalty_w_per_lb = weight_penalty_w_per_lb


def sample_random_inputs():
    """Sample one set of nominal environmental and mission inputs."""
    wind_mph = random.uniform(0.0, 25.0)
    temp_f = random.uniform(32.0, 95.0)
    payload_lb = random.uniform(0.0, 5.0)
    distance_factor = random.uniform(0.9, 1.1)
    return wind_mph, temp_f, payload_lb, distance_factor


def sample_extreme_inputs():
    """Sample one set of extreme / worst-case-biased inputs."""
    wind_mph = random.triangular(10.0, 20.0, 20.0)
    temp_f = random.triangular(10.0, 40.0, 10.0)
    payload_lb = random.triangular(4.0, 5.0, 5.0)
    distance_factor = random.triangular(1.0, 1.1, 1.1)
    return wind_mph, temp_f, payload_lb, distance_factor


def effective_capacity_wh(battery: BatteryConfig, temp_f: float) -> float:
    """Compute effective capacity (Wh) as a function of temperature."""
    if temp_f >= 70.0:
        return battery.capacity_wh

    delta = 70.0 - temp_f
    penalty = battery.cold_capacity_penalty_per_deg * delta
    penalty = min(max(penalty, 0.0), 0.5)
    return battery.capacity_wh * (1.0 - penalty)


def simulate_mission(battery: BatteryConfig, mission: MissionConfig, rng_inputs=None):
    """Simulate one mission trial for a given battery under sampled conditions."""
    if rng_inputs is None:
        wind_mph, temp_f, payload_lb, distance_factor = sample_random_inputs()
    else:
        wind_mph, temp_f, payload_lb, distance_factor = rng_inputs

    distance_mi = mission.nominal_distance_mi * distance_factor
    total_takeoff_weight_lb = mission.empty_drone_weight_lb + payload_lb

    power_w = (
        battery.base_power_w
        + mission.weight_penalty_w_per_lb * mission.empty_drone_weight_lb
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
    energy_remaining_pct = 100.0 * energy_remaining_wh / cap_eff_wh if cap_eff_wh > 0 else 0.0

    return {
        "battery": battery.label,
        "wind_mph": wind_mph,
        "temp_f": temp_f,
        "payload_lb": payload_lb,
        "distance_mi": distance_mi,
        "empty_drone_weight_lb": mission.empty_drone_weight_lb,
        "total_takeoff_weight_lb": total_takeoff_weight_lb,
        "cap_eff_wh": cap_eff_wh,
        "energy_required_wh": energy_required_wh,
        "energy_available_for_mission_wh": energy_available_for_mission_wh,
        "energy_remaining_wh": energy_remaining_wh,
        "energy_remaining_pct": energy_remaining_pct,
        "success": success,
    }


def process_generator(num_runs: int, batteries, mission: MissionConfig):
    """Yield one nominal simulation record at a time."""
    for _ in range(num_runs):
        for battery in batteries:
            rng_inputs = sample_random_inputs()
            yield simulate_mission(battery, mission, rng_inputs=rng_inputs)


def process_generator_extreme(num_runs: int, batteries, mission: MissionConfig):
    """Yield one extreme-biased simulation record at a time."""
    for _ in range(num_runs):
        for battery in batteries:
            rng_inputs = sample_extreme_inputs()
            yield simulate_mission(battery, mission, rng_inputs=rng_inputs)


def run_monte_carlo(num_runs: int, batteries, mission: MissionConfig):
    """Run nominal Monte Carlo and collect results per battery label."""
    results = {b.label: [] for b in batteries}
    for record in process_generator(num_runs, batteries, mission):
        results[record["battery"]].append(record)
    return results


def run_monte_carlo_extreme(num_runs: int, batteries, mission: MissionConfig):
    """Run extreme Monte Carlo and collect results per battery label."""
    results = {b.label: [] for b in batteries}
    for record in process_generator_extreme(num_runs, batteries, mission):
        results[record["battery"]].append(record)
    return results


def percentile(data, p):
    """Compute percentile using linear interpolation."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    n_local = len(data_sorted)
    if n_local == 1:
        return data_sorted[0]
    idx = (p / 100.0) * (n_local - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return data_sorted[int(idx)]
    frac = idx - lower
    return data_sorted[lower] * (1 - frac) + data_sorted[upper] * frac


def summarize_results(records, battery_config: BatteryConfig, mission_config: MissionConfig):
    """Compute summary statistics for one battery's records."""
    if not records:
        return None

    remaining_pct = [r["energy_remaining_pct"] for r in records]
    remaining_wh = [r["energy_remaining_wh"] for r in records]
    successes = [1 if r["success"] else 0 for r in records]
    miles = [r["distance_mi"] for r in records]
    payloads = [r["payload_lb"] for r in records]
    wh_used_list = [r["energy_required_wh"] for r in records]
    cap_eff_list = [r["cap_eff_wh"] for r in records]
    successful_records = [r for r in records if r["success"]]

    avg_wh_per_trip = statistics.mean(wh_used_list) if wh_used_list else 0.0
    avg_cap_eff = statistics.mean(cap_eff_list) if cap_eff_list else 0.0
    avg_usable_wh = avg_cap_eff * (1.0 - mission_config.safety_energy_fraction)

    missions_per_charge = (avg_usable_wh / avg_wh_per_trip) if avg_wh_per_trip > 0 else 0.0
    deliveries_per_charge = math.floor(missions_per_charge)

    n = len(records)
    mean_remaining = statistics.mean(remaining_pct) if remaining_pct else 0.0
    p5_remaining = percentile(remaining_pct, 5)
    p95_remaining = percentile(remaining_pct, 95)
    success_rate = (sum(successes) / n * 100.0) if n > 0 else 0.0

    avg_miles_all = statistics.mean(miles) if miles else 0.0
    avg_payload_all = statistics.mean(payloads) if payloads else 0.0
    avg_remaining_wh = statistics.mean(remaining_wh) if remaining_wh else 0.0

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
        "avg_wh_per_trip": avg_wh_per_trip,
        "avg_cap_eff_wh": avg_cap_eff,
        "avg_usable_wh": avg_usable_wh,
        "avg_remaining_wh": avg_remaining_wh,
        "deliveries_per_charge": deliveries_per_charge,
        "missions_per_charge": missions_per_charge,
        "total_runs": n,
        "successful_runs": sum(successes),
        "best_case_wh": max(wh_used_list) if wh_used_list else 0.0,
        "worst_case_wh": min(wh_used_list) if wh_used_list else 0.0,
        "median_wh_per_trip": percentile(wh_used_list, 50),
    }


# -----------------------------
# GUI
# -----------------------------

class MonteCarloGUI:
    """Tkinter GUI for running nominal/extreme Monte Carlo simulations."""

    def __init__(self, root):
        self.root = root
        self.root.title("Drone Battery Monte Carlo Process Generator")
        self.root.geometry("1360x940")
        self.root.minsize(1180, 820)

        self.battery_a = BatteryConfig(
            key="A",
            label="Li-ion (NMC, cylindrical)",
            capacity_wh=1300.0,
            base_power_w=180.0,
            payload_penalty_w_per_lb=10.0,
            wind_penalty_w_per_mph=1.5,
            cold_capacity_penalty_per_deg=0.0050,
            cold_power_penalty_per_deg=0.0025,
        )

        # Chemistry-aligned assumptions:
        # - Li-ion carries more nominal energy, but suffers less from cold derating.
        # - LiPo is modeled as a high-power pouch pack with lower nominal energy,
        #   slightly lower nominal propulsion burden, and harsher cold derating.
        self.battery_b = BatteryConfig(
            key="B",
            label="LiPo (pouch, high-power)",
            capacity_wh=1100.0,
            base_power_w=170.0,
            payload_penalty_w_per_lb=10.5,
            wind_penalty_w_per_mph=1.3,
            cold_capacity_penalty_per_deg=0.0080,
            cold_power_penalty_per_deg=0.0040,
        )

        self.mission = MissionConfig(
            nominal_distance_mi=15.0,
            cruise_speed_mph=25.0,
            safety_energy_fraction=0.20,
            empty_drone_weight_lb=50.0,
            weight_penalty_w_per_lb=1.0,
        )

        self.batteries = [self.battery_a, self.battery_b]

        self.figure = None
        self.canvas = None
        self.extreme_figure = None
        self.extreme_canvas = None
        self.summary_sections = {}
        self.extreme_summary_sections = {}
        self.summary_content_frame = None
        self.extreme_summary_content_frame = None
        self.plot_container = None
        self.extreme_plot_container = None
        self.nominal_kpi_vars = {}
        self.extreme_kpi_vars = {}

        self._configure_styles()
        self._build_tabs()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Custom.TNotebook.Tab",
            background="#4A6FFF",
            foreground="white",
            padding=[14, 8],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", "#2E47B8")],
            foreground=[("selected", "white")],
        )
        style.configure("Section.TLabelframe", padding=8)
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("CardTitle.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("CardNote.TLabel", font=("Segoe UI", 9))

    def _build_tabs(self):
        notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        notebook.pack(fill="both", expand=True)

        sim_tab = ttk.Frame(notebook)
        notebook.add(sim_tab, text="Simulation")
        self._build_widgets(sim_tab, extreme=False)

        params_tab = ttk.Frame(notebook)
        notebook.add(params_tab, text="Parameters Summary")

        extreme_tab = ttk.Frame(notebook)
        notebook.add(extreme_tab, text="Extreme Conditions")
        self._build_widgets(extreme_tab, extreme=True)

        params_text = tk.Text(params_tab, wrap="word", font=("Consolas", 11))
        params_text.pack(fill="both", expand=True, padx=10, pady=10)
        params_text.insert(tk.END, self._generate_parameters_summary())
        params_text.configure(state="disabled")

    def _build_widgets(self, parent, extreme: bool):
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(
            main_frame,
            text="Extreme Simulation Settings" if extreme else "Simulation Settings",
            style="Section.TLabelframe",
        )
        controls.pack(side="top", fill="x", pady=(0, 8))

        ttk.Label(controls, text="Number of runs per battery:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        if extreme:
            self.extreme_runs_var = tk.StringVar(value="500")
            ttk.Entry(controls, textvariable=self.extreme_runs_var, width=10).grid(
                row=0, column=1, sticky="w", padx=5, pady=5
            )
            ttk.Button(
                controls,
                text="Run Extreme Simulation",
                command=self.run_extreme_simulation,
            ).grid(row=0, column=2, sticky="w", padx=10, pady=5)
            ttk.Label(
                controls,
                text="Extreme sampling biases wind high, temp low, payload high, and distance high.",
            ).grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        else:
            self.num_runs_var = tk.StringVar(value="500")
            ttk.Entry(controls, textvariable=self.num_runs_var, width=10).grid(
                row=0, column=1, sticky="w", padx=5, pady=5
            )
            ttk.Button(
                controls,
                text="Run Simulation",
                command=self.run_simulation,
            ).grid(row=0, column=2, sticky="w", padx=10, pady=5)

        kpi_frame = ttk.LabelFrame(
            main_frame,
            text="Quick Comparison" if not extreme else "Extreme Quick Comparison",
            style="Section.TLabelframe",
        )
        kpi_frame.pack(side="top", fill="x", pady=(0, 8))
        if extreme:
            self.extreme_kpi_vars = self._build_kpi_cards(kpi_frame)
        else:
            self.nominal_kpi_vars = self._build_kpi_cards(kpi_frame)

        paned = ttk.PanedWindow(main_frame, orient="vertical")
        paned.pack(side="top", fill="both", expand=True)

        summary_frame = ttk.LabelFrame(
            paned,
            text="Summary Details",
            style="Section.TLabelframe",
        )
        plot_frame = ttk.LabelFrame(
            paned,
            text="Distributions",
            style="Section.TLabelframe",
        )

        paned.add(summary_frame, weight=3)
        paned.add(plot_frame, weight=4)

        content_frame, sections_dict = self._create_scrollable_summary_area(summary_frame)

        if extreme:
            self.extreme_summary_content_frame = content_frame
            self.extreme_summary_sections = sections_dict
            self.extreme_plot_container = plot_frame
        else:
            self.summary_content_frame = content_frame
            self.summary_sections = sections_dict
            self.plot_container = plot_frame


    def _create_scrollable_summary_area(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _resize_content(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        canvas.bind("<Configure>", _resize_content)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return content, {}

    def _clear_summary_boxes(self, content_frame, sections_dict):
        if content_frame is None:
            return
        for child in content_frame.winfo_children():
            child.destroy()
        sections_dict.clear()

    def _create_summary_box(self, parent, title, body, row, column, columnspan=1):
        box = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe")
        box.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=6, pady=6)
        text = tk.Text(box, height=8, wrap="word", font=("Consolas", 10), relief="flat")
        text.pack(fill="both", expand=True, padx=6, pady=6)
        text.insert("1.0", body)
        text.configure(state="disabled")
        return box

    def _build_battery_summary_sections(self, battery, summary, records, title_suffix):
        last = records[-1]
        header = f"{battery.label} — {title_suffix}"

        sections = [
            (
                header,
                f"Energy used this run:            {last['energy_required_wh']:.1f} Wh\n"
                f"Effective capacity this run:     {last['cap_eff_wh']:.1f} Wh\n"
                f"Energy available after reserve:  {last['energy_available_for_mission_wh']:.1f} Wh\n"
                f"Remaining this run:              {last['energy_remaining_pct']:.1f}%\n"
                f"Inputs: wind={last['wind_mph']:.1f} mph, temp={last['temp_f']:.1f}°F, "
                f"payload={last['payload_lb']:.2f} lb, takeoff wt={last['total_takeoff_weight_lb']:.2f} lb, "
                f"miles={last['distance_mi']:.2f}"
            ),
            (
                "Run Summary",
                f"Total simulation trials:         {summary['total_runs']}\n"
                f"Successful runs:                 {summary['successful_runs']} / {summary['total_runs']} "
                f"({summary['success_rate']:.1f}%)"
            ),
            (
                "Remaining Energy Distribution",
                f"Mean remaining energy:           {summary['mean_remaining']:.1f}%\n"
                f"5th percentile remaining:        {summary['p5_remaining']:.1f}%\n"
                f"95th percentile remaining:       {summary['p95_remaining']:.1f}%\n"
                f"Average remaining energy:        {summary['avg_remaining_wh']:.1f} Wh"
            ),
            (
                "Operational Averages",
                f"Avg miles (all trials):          {summary['avg_miles_all']:.2f} mi\n"
                f"Avg payload (all trials):        {summary['avg_payload_all']:.2f} lb\n"
                f"Avg miles (successful only):     {summary['avg_miles_success']:.2f} mi\n"
                f"Avg payload (successful only):   {summary['avg_payload_success']:.2f} lb"
            ),
            (
                "Energy Use & Deliveries",
                f"Avg energy used per mission:     {summary['avg_wh_per_trip']:.1f} Wh\n"
                f"Median energy used per mission:  {summary['median_wh_per_trip']:.1f} Wh\n"
                f"Avg effective capacity:          {summary['avg_cap_eff_wh']:.1f} Wh\n"
                f"Avg usable energy:               {summary['avg_usable_wh']:.1f} Wh\n"
                f"Missions per charge (float):     {summary['missions_per_charge']:.2f}\n"
                f"Approx. deliveries/full charge:  {summary['deliveries_per_charge']}"
            ),
        ]
        return sections

    def _render_summary_boxes(self, extreme, summaries, results):
        content_frame = self.extreme_summary_content_frame if extreme else self.summary_content_frame
        sections_dict = self.extreme_summary_sections if extreme else self.summary_sections
        self._clear_summary_boxes(content_frame, sections_dict)

        if content_frame is None:
            return

        for col in range(2):
            content_frame.columnconfigure(col, weight=1)

        row = 0
        title_suffix = "EXTREME Monte Carlo Results" if extreme else "Monte Carlo Results"

        for battery in self.batteries:
            recs = results.get(battery.label, [])
            summary = summaries.get(battery.label)
            if not recs or not summary:
                self._create_summary_box(content_frame, battery.label, "No records generated.", row, 0, columnspan=2)
                row += 1
                continue

            section_pairs = self._build_battery_summary_sections(battery, summary, recs, title_suffix)
            self._create_summary_box(content_frame, section_pairs[0][0], section_pairs[0][1], row, 0, columnspan=2)
            row += 1
            self._create_summary_box(content_frame, section_pairs[1][0], section_pairs[1][1], row, 0)
            self._create_summary_box(content_frame, section_pairs[2][0], section_pairs[2][1], row, 1)
            row += 1
            self._create_summary_box(content_frame, section_pairs[3][0], section_pairs[3][1], row, 0)
            self._create_summary_box(content_frame, section_pairs[4][0], section_pairs[4][1], row, 1)
            row += 1

        if len(self.batteries) >= 2:
            a_name = self.batteries[0].label
            b_name = self.batteries[1].label
            if a_name in summaries and b_name in summaries:
                A = summaries[a_name]
                B = summaries[b_name]
                comparison_label = "Li-ion vs LiPo Comparison (Extreme)" if extreme else "Li-ion vs LiPo Comparison (Nominal)"
                comparison_body = (
                    f"Δ Mean remaining (Li-ion - LiPo):       {A['mean_remaining'] - B['mean_remaining']:.2f} %\n"
                    f"Δ 5th pct remaining (Li-ion - LiPo):    {A['p5_remaining'] - B['p5_remaining']:.2f} %\n"
                    f"Δ 95th pct remaining (Li-ion - LiPo):   {A['p95_remaining'] - B['p95_remaining']:.2f} %\n"
                    f"Δ Avg Wh/mission (Li-ion - LiPo):       {A['avg_wh_per_trip'] - B['avg_wh_per_trip']:.2f} Wh\n"
                    f"Δ Missions/charge (Li-ion - LiPo):      {A['missions_per_charge'] - B['missions_per_charge']:.2f}\n"
                    f"Li-ion success rate:                    {A['success_rate']:.1f}%\n"
                    f"LiPo success rate:                      {B['success_rate']:.1f}%"
                )
                self._create_summary_box(content_frame, comparison_label, comparison_body, row, 0, columnspan=2)

    def _build_kpi_cards(self, parent):
        keys = [
            ("liion_success", "Li-ion success"),
            ("lipo_success", "LiPo success"),
            ("liion_missions", "Li-ion missions/charge"),
            ("lipo_missions", "LiPo missions/charge"),
            ("winner", "Higher missions/charge"),
            ("energy_winner", "Lower avg Wh/mission"),
        ]
        vars_dict = {}
        for idx, (key, title) in enumerate(keys):
            card = ttk.Frame(parent, relief="ridge", borderwidth=1, padding=8)
            card.grid(row=0, column=idx, sticky="nsew", padx=4, pady=4)
            parent.columnconfigure(idx, weight=1)
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            value_var = tk.StringVar(value="—")
            note_var = tk.StringVar(value="")
            ttk.Label(card, textvariable=value_var, style="CardValue.TLabel").pack(anchor="w", pady=(4, 2))
            ttk.Label(card, textvariable=note_var, style="CardNote.TLabel").pack(anchor="w")
            vars_dict[key] = (value_var, note_var)
        return vars_dict

    def _set_card(self, card_vars, key, value, note=""):
        value_var, note_var = card_vars[key]
        value_var.set(value)
        note_var.set(note)

    def _clear_kpis(self, card_vars):
        for key in card_vars:
            self._set_card(card_vars, key, "—", "")

    def _winner_by_metric(self, a_label, a_value, b_label, b_value, higher_is_better=True):
        if abs(a_value - b_value) < 1e-9:
            return "Tie"
        if higher_is_better:
            return a_label if a_value > b_value else b_label
        return a_label if a_value < b_value else b_label

    def _update_kpi_cards(self, summaries, card_vars):
        if len(self.batteries) < 2:
            self._clear_kpis(card_vars)
            return

        a_name = self.batteries[0].label
        b_name = self.batteries[1].label
        if a_name not in summaries or b_name not in summaries:
            self._clear_kpis(card_vars)
            return

        A = summaries[a_name]
        B = summaries[b_name]

        self._set_card(card_vars, "liion_success", f"{A['success_rate']:.1f}%", f"{A['successful_runs']}/{A['total_runs']} runs")
        self._set_card(card_vars, "lipo_success", f"{B['success_rate']:.1f}%", f"{B['successful_runs']}/{B['total_runs']} runs")
        self._set_card(card_vars, "liion_missions", f"{A['missions_per_charge']:.2f}", f"whole missions = {A['deliveries_per_charge']}")
        self._set_card(card_vars, "lipo_missions", f"{B['missions_per_charge']:.2f}", f"whole missions = {B['deliveries_per_charge']}")

        winner = self._winner_by_metric("Li-ion", A["missions_per_charge"], "LiPo", B["missions_per_charge"], True)
        delta_missions = abs(A["missions_per_charge"] - B["missions_per_charge"])
        self._set_card(card_vars, "winner", winner, f"Δ = {delta_missions:.2f} missions")

        energy_winner = self._winner_by_metric("Li-ion", A["avg_wh_per_trip"], "LiPo", B["avg_wh_per_trip"], False)
        delta_energy = abs(A["avg_wh_per_trip"] - B["avg_wh_per_trip"])
        self._set_card(card_vars, "energy_winner", energy_winner, f"Δ = {delta_energy:.1f} Wh per trip")

    def _format_battery_summary(self, battery, summary, records, title_suffix):
        last = records[-1]
        lines = [
            f"{battery.label} — {title_suffix}",
            "=" * (len(battery.label) + len(title_suffix) + 3),
            "",
            "Example single run (most recent)",
            "-------------------------------",
            f"Energy used this run:            {last['energy_required_wh']:.1f} Wh",
            f"Effective capacity this run:     {last['cap_eff_wh']:.1f} Wh",
            f"Energy available after reserve:  {last['energy_available_for_mission_wh']:.1f} Wh",
            f"Remaining this run:              {last['energy_remaining_pct']:.1f}%",
            f"Inputs: wind={last['wind_mph']:.1f} mph, temp={last['temp_f']:.1f}°F, "
            f"payload={last['payload_lb']:.2f} lb, takeoff wt={last['total_takeoff_weight_lb']:.2f} lb, "
            f"miles={last['distance_mi']:.2f}",
            "",
            "Run Summary",
            "-----------",
            f"Total simulation trials:         {summary['total_runs']}",
            f"Successful runs:                 {summary['successful_runs']} / {summary['total_runs']} ({summary['success_rate']:.1f}%)",
            "",
            "Remaining Energy Distribution",
            "-----------------------------",
            f"Mean remaining energy:           {summary['mean_remaining']:.1f}%",
            f"5th percentile remaining:        {summary['p5_remaining']:.1f}%",
            f"95th percentile remaining:       {summary['p95_remaining']:.1f}%",
            f"Average remaining energy:        {summary['avg_remaining_wh']:.1f} Wh",
            "",
            "Operational Averages",
            "--------------------",
            f"Avg miles (all trials):          {summary['avg_miles_all']:.2f} mi",
            f"Avg payload (all trials):        {summary['avg_payload_all']:.2f} lb",
            f"Avg miles (successful only):     {summary['avg_miles_success']:.2f} mi",
            f"Avg payload (successful only):   {summary['avg_payload_success']:.2f} lb",
            "",
            "Energy Use & Deliveries",
            "-----------------------",
            f"Avg energy used per mission:     {summary['avg_wh_per_trip']:.1f} Wh",
            f"Median energy used per mission:  {summary['median_wh_per_trip']:.1f} Wh",
            f"Avg effective capacity:          {summary['avg_cap_eff_wh']:.1f} Wh",
            f"Avg usable energy:               {summary['avg_usable_wh']:.1f} Wh",
            f"Missions per charge (float):     {summary['missions_per_charge']:.2f}",
            f"Approx. deliveries/full charge:  {summary['deliveries_per_charge']}",
            "",
        ]
        return "\n".join(lines)

    def _append_comparison_block(self, text_widget, summaries, label):
        if len(self.batteries) < 2:
            return
        a_name = self.batteries[0].label
        b_name = self.batteries[1].label
        if a_name not in summaries or b_name not in summaries:
            return

        A = summaries[a_name]
        B = summaries[b_name]
        text_widget.insert(
            tk.END,
            f"{label}\n"
            f"{'=' * len(label)}\n\n"
            f"Δ Mean remaining (Li-ion - LiPo):       {A['mean_remaining'] - B['mean_remaining']:.2f} %\n"
            f"Δ 5th pct remaining (Li-ion - LiPo):    {A['p5_remaining'] - B['p5_remaining']:.2f} %\n"
            f"Δ 95th pct remaining (Li-ion - LiPo):   {A['p95_remaining'] - B['p95_remaining']:.2f} %\n"
            f"Δ Avg Wh/mission (Li-ion - LiPo):       {A['avg_wh_per_trip'] - B['avg_wh_per_trip']:.2f} Wh\n"
            f"Δ Missions/charge (Li-ion - LiPo):      {A['missions_per_charge'] - B['missions_per_charge']:.2f}\n"
            f"Li-ion success rate:                    {A['success_rate']:.1f}%\n"
            f"LiPo success rate:                      {B['success_rate']:.1f}%\n"
        )

    def _run_common(self, extreme: bool):
        card_vars = self.extreme_kpi_vars if extreme else self.nominal_kpi_vars

        try:
            num_runs = int(self.extreme_runs_var.get() if extreme else self.num_runs_var.get())
            if num_runs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of runs must be a positive integer.")
            return

        results = (
            run_monte_carlo_extreme(num_runs, self.batteries, self.mission)
            if extreme
            else run_monte_carlo(num_runs, self.batteries, self.mission)
        )

        summaries = {}

        for battery in self.batteries:
            recs = results.get(battery.label, [])
            if not recs:
                continue

            summary = summarize_results(recs, battery_config=battery, mission_config=self.mission)
            if not summary:
                continue

            summaries[battery.label] = summary

        self._render_summary_boxes(extreme, summaries, results)
        self._update_kpi_cards(summaries, card_vars)
        self._plot_distributions(results, extreme=extreme)

    def run_simulation(self):
        self._run_common(extreme=False)

    def run_extreme_simulation(self):
        self._run_common(extreme=True)

    def _plot_distributions(self, results, extreme: bool = False):
        container = self.extreme_plot_container if extreme else self.plot_container
        existing_canvas = self.extreme_canvas if extreme else self.canvas

        if existing_canvas is not None:
            existing_canvas.get_tk_widget().destroy()
            if extreme:
                self.extreme_canvas = None
            else:
                self.canvas = None

        figure = Figure(figsize=(10, 5.5))
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)

        for battery in self.batteries:
            remaining = [r["energy_remaining_pct"] for r in results[battery.label]]
            wh_used = [r["energy_required_wh"] for r in results[battery.label]]
            ax1.hist(remaining, bins=20, alpha=0.55, label=battery.label)
            ax2.hist(wh_used, bins=20, alpha=0.55, label=battery.label)

        ax1.set_xlabel("Remaining Energy (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Remaining Energy Distribution")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=9)

        ax2.set_xlabel("Trip Energy (Wh)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Trip Energy Distribution")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=9)

        main_title = "EXTREME Monte Carlo Battery Results" if extreme else "Monte Carlo Battery Results"
        figure.suptitle(main_title, fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.96])

        canvas = FigureCanvasTkAgg(figure, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        if extreme:
            self.extreme_figure = figure
            self.extreme_canvas = canvas
        else:
            self.figure = figure
            self.canvas = canvas

    def _generate_parameters_summary(self):
        usable_nominal_a = self.battery_a.capacity_wh * (1.0 - self.mission.safety_energy_fraction)
        usable_nominal_b = self.battery_b.capacity_wh * (1.0 - self.mission.safety_energy_fraction)
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
            "  Temperature:     10–40 °F (cold-biased)\n"
            "  Payload:         4–5 lb (high-biased)\n"
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
            f"  Nominal usable:  {usable_nominal_a:.1f} Wh\n"
            f"  Base power:      {self.battery_a.base_power_w} W\n"
            f"  Payload penalty: {self.battery_a.payload_penalty_w_per_lb} W/lb\n"
            f"  Wind penalty:    {self.battery_a.wind_penalty_w_per_mph} W/mph\n"
            f"  Cold capacity:   {self.battery_a.cold_capacity_penalty_per_deg} per °F < 70\n"
            f"  Cold power:      {self.battery_a.cold_power_penalty_per_deg} per °F < 70\n\n"
            f"{self.battery_b.label}:\n"
            f"  Capacity:        {self.battery_b.capacity_wh} Wh\n"
            f"  Nominal usable:  {usable_nominal_b:.1f} Wh\n"
            f"  Base power:      {self.battery_b.base_power_w} W\n"
            f"  Payload penalty: {self.battery_b.payload_penalty_w_per_lb} W/lb\n"
            f"  Wind penalty:    {self.battery_b.wind_penalty_w_per_mph} W/mph\n"
            f"  Cold capacity:   {self.battery_b.cold_capacity_penalty_per_deg} per °F < 70\n"
            f"  Cold power:      {self.battery_b.cold_power_penalty_per_deg} per °F < 70\n\n"
            "Notes\n"
            "-----\n"
            f"- {self.battery_a.label} is modeled at {self.battery_a.capacity_wh:.0f} Wh nominal capacity.\n"
            f"- {self.battery_b.label} is modeled at {self.battery_b.capacity_wh:.0f} Wh nominal capacity.\n"
            "- Li-ion is modeled as the higher-energy chemistry; LiPo is modeled as the higher-power but lower-energy chemistry.\n"
            "- LiPo has harsher cold derating than Li-ion in both effective capacity and power multiplier terms.\n"
            "- The 20% reserve is mission-wide and applies to both batteries.\n"
            "- Effective capacity is recalculated independently for each run from that run's temperature only.\n"
            "- For temperatures at or above 70°F, the model returns full nominal battery capacity.\n"
        )


def generate_docs():
    """Execute the Doxygen command to refresh documentation."""
    try:
        subprocess.run(["doxygen", "Doxyfile"], check=True)
        print("Documentation generated successfully.")
    except FileNotFoundError:
        print("Error: Doxygen not found in system path.")



def main():
    root = tk.Tk()
    app = MonteCarloGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    generate_docs()
