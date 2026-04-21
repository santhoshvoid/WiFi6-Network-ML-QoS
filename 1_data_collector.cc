/* =============================================================
 * 1_data_collector.cc  — MIXED TRAFFIC VERSION
 *
 * PURPOSE:
 *   Run ns-3 Wi-Fi 6 simulations with MIXED traffic
 *   (VoIP + Video + HTTP + VPN + IoT simultaneously)
 *   under different RU, MCS, and station density combinations.
 *   Output: ns3_training_data.csv
 *
 * HOW TRAFFIC IS MIXED:
 *   Each simulation has nSta stations.
 *   Stations are assigned traffic types in a round-robin:
 *     Station 0 → VoIP
 *     Station 1 → Video
 *     Station 2 → HTTP
 *     Station 3 → VPN
 *     Station 4 → IoT
 *     Station 5 → VoIP  (repeats)
 *     ...
 *   All types transmit simultaneously to the AP.
 *   FlowMonitor records per-flow stats so each traffic
 *   type's experience is measured separately.
 *
 * PARAMETER SWEEP:
 *   RU   : {1, 3, 5, 8, 12, 18, 25, 37}   — 8 values
 *   MCS  : {0, 3, 6, 9, 11}               — 5 values
 *   nSta : {5, 10, 15, 20}                — 4 values
 *   → 8 × 5 × 4 = 160 scenarios
 *   Each scenario has nSta flows → ~1600+ CSV rows
 *
 * BUILD & RUN:
 *   cp 1_data_collector.cc ~/ns-3-dev/scratch/
 *   cd ~/ns-3-dev && ./ns3 build
 *   ./ns3 run scratch/1_data_collector
 * ============================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/applications-module.h"

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WifiQoSDataCollector");

// ── Traffic type definitions ──────────────────────────────────
struct TrafficDef {
    std::string name;
    uint32_t    pktSize;    // bytes
    int         intervalMs; // ms between packets
    int         priority;   // ground truth label (0=Low → 3=High)
    int         twtMs;      // ground truth TWT interval
};

// Five traffic types — assigned round-robin to stations
static const std::vector<TrafficDef> TRAFFIC_TYPES = {
    {"VoIP",  160,  20,   3,   8},
    {"Video", 1400, 40,   2,  20},
    {"HTTP",  800,  100,  1,  55},
    {"VPN",   900,  80,   1,  60},
    {"IoT",   64,   1000, 0, 280},
};

// ── Parameter sweep ───────────────────────────────────────────
static const std::vector<int> RU_VALUES   = {1,3,5,8,12,18,25,37};
static const std::vector<int> MCS_VALUES  = {0,3,6,9,11};
static const std::vector<int> STA_VALUES  = {5,10,15,20};

// ── Per-flow result ───────────────────────────────────────────
struct FlowResult {
    std::string trafficType;
    uint32_t    pktSize;
    int         intervalMs;
    int         numStations;
    int         ru;
    int         mcs;
    double      throughput_mbps;
    double      mean_delay_ms;
    double      mean_jitter_ms;
    double      packet_loss_rate;
    int         priority;
    int         twtMs;
};

// ── Run one mixed-traffic scenario ───────────────────────────
// Returns one FlowResult per station (per traffic type present)
std::vector<FlowResult> RunMixedScenario(
    int    nSta,
    int    ru,
    int    mcs,
    double simDur)
{
    std::vector<FlowResult> results;

    // ── Nodes ─────────────────────────────────────────────────
    NodeContainer staNodes, apNode;
    staNodes.Create(nSta);
    apNode.Create(1);

    // ── Channel ───────────────────────────────────────────────
    YansWifiChannelHelper channel =
        YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("ChannelSettings",
            StringValue("{0, 80, BAND_5GHZ, 0}"));

    // ── WiFi 6 ────────────────────────────────────────────────
    Ssid ssid("collect");
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    WifiMacHelper macAp;
    macAp.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDev =
        wifi.Install(phy, macAp, apNode);

    WifiMacHelper macSta;
    macSta.SetType("ns3::StaWifiMac",
                   "Ssid",          SsidValue(ssid),
                   "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDev =
        wifi.Install(phy, macSta, staNodes);

    // ── Mobility — grid, all within 5 m ───────────────────────
    MobilityHelper mob;
    mob.SetPositionAllocator(
        "ns3::GridPositionAllocator",
        "MinX",      DoubleValue(1.0),
        "MinY",      DoubleValue(1.0),
        "DeltaX",    DoubleValue(2.0),
        "DeltaY",    DoubleValue(2.0),
        "GridWidth", UintegerValue(5));
    mob.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");
    mob.Install(staNodes);

    MobilityHelper apMob;
    apMob.SetPositionAllocator(
        "ns3::GridPositionAllocator",
        "MinX", DoubleValue(0.0),
        "MinY", DoubleValue(0.0));
    apMob.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");
    apMob.Install(apNode);

    // ── Internet stack + IPs ──────────────────────────────────
    InternetStackHelper internet;
    internet.Install(apNode);
    internet.Install(staNodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apIface = ipv4.Assign(apDev);
    ipv4.Assign(staDev);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ── Sink on AP ────────────────────────────────────────────
    uint16_t port = 9;
    PacketSinkHelper sinkH(
        "ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp =
        sinkH.Install(apNode.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(simDur));

    // ── FlowMonitor ───────────────────────────────────────────
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // ── Install MIXED traffic — one type per station ──────────
    // Round-robin assignment: VoIP, Video, HTTP, VPN, IoT, VoIP...
    // All stations transmit SIMULTANEOUSLY
    for (int i = 0; i < nSta; i++)
    {
        // Pick traffic type for this station
        const TrafficDef& tr =
            TRAFFIC_TYPES[i % TRAFFIC_TYPES.size()];

        // Effective send interval based on RU
        double ruRateMbps =
            ((double)ru / 37.0) * 600.0;
        double minIntMs =
            (tr.pktSize * 8.0)
            / (ruRateMbps * 1e6) * 1000.0;
        int effIntMs = std::max(
            tr.intervalMs, (int)std::ceil(minIntMs));

        // App rate
        double bps = (double)tr.pktSize * 8.0
                     / (double)effIntMs * 1000.0;
        std::ostringstream rateStr;
        rateStr << (uint64_t)bps << "bps";

        // Stagger start times slightly so not all
        // stations associate at exactly the same instant
        double startTime = 2.0 + i * 0.05;

        OnOffHelper onoff(
            "ns3::UdpSocketFactory",
            InetSocketAddress(
                apIface.GetAddress(0), port));
        onoff.SetConstantRate(
            DataRate(rateStr.str()), tr.pktSize);
        onoff.SetAttribute("OnTime",  StringValue(
            "ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue(
            "ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer app =
            onoff.Install(staNodes.Get(i));
        app.Start(Seconds(startTime));
        app.Stop(Seconds(simDur - 1.0));
    }

    // ── Run ───────────────────────────────────────────────────
    Simulator::Stop(Seconds(simDur));
    Simulator::Run();

    // ── Extract per-flow stats ────────────────────────────────
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> clf =
        DynamicCast<Ipv4FlowClassifier>(
            flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats =
        monitor->GetFlowStats();

    for (auto& kv : stats)
    {
        auto& s = kv.second;
        if (s.txPackets == 0) continue;

        double dur = s.rxPackets > 0
            ? s.timeLastRxPacket.GetSeconds()
              - s.timeFirstTxPacket.GetSeconds()
            : 0.0;

        double tput   = dur > 0
            ? s.rxBytes * 8.0 / dur / 1e6 : 0.0;
        double delay  = s.rxPackets > 0
            ? s.delaySum.GetSeconds()
              / s.rxPackets * 1000.0 : 0.0;
        double jitter = s.rxPackets > 1
            ? s.jitterSum.GetSeconds()
              / (s.rxPackets - 1) * 1000.0 : 0.0;
        double loss   =
            (double)(s.txPackets - s.rxPackets)
            / s.txPackets;

        // Map flow ID back to station index
        // FlowMonitor IDs start at 1
        uint32_t staIdx = (kv.first - 1)
                          % (uint32_t)nSta;
        const TrafficDef& tr =
            TRAFFIC_TYPES[staIdx % TRAFFIC_TYPES.size()];

        // Effective interval for this traffic + RU
        double ruRateMbps =
            ((double)ru / 37.0) * 600.0;
        double minIntMs =
            (tr.pktSize * 8.0)
            / (ruRateMbps * 1e6) * 1000.0;
        int effIntMs = std::max(
            tr.intervalMs, (int)std::ceil(minIntMs));

        FlowResult r;
        r.trafficType      = tr.name;
        r.pktSize          = tr.pktSize;
        r.intervalMs       = effIntMs;   // effective interval
        r.numStations      = nSta;
        r.ru               = ru;
        r.mcs              = mcs;
        r.throughput_mbps  = tput;
        r.mean_delay_ms    = delay;
        r.mean_jitter_ms   = jitter;
        r.packet_loss_rate = loss;
        r.priority         = tr.priority;
        r.twtMs            = tr.twtMs;

        results.push_back(r);
    }

    Simulator::Destroy();
    return results;
}

// ── main ──────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // Suppress ns-3 log noise
    LogComponentDisableAll(LOG_LEVEL_ALL);

    std::string outFile = "ns3_training_data.csv";
    std::ofstream csv(outFile);

    // CSV header — must match FEATURE_COLS in 2_train_model.py
    csv << "traffic_type,"
        << "packet_size,"
        << "interval_ms,"
        << "num_stations,"
        << "ru,"
        << "mcs,"
        << "throughput_mbps,"
        << "mean_delay_ms,"
        << "mean_jitter_ms,"
        << "packet_loss_rate,"
        << "priority,"
        << "twt_ms\n";

    int totalScenarios =
        (int)(RU_VALUES.size()
              * MCS_VALUES.size()
              * STA_VALUES.size());
    int scenarioDone = 0;
    int rowsWritten  = 0;

    std::cout << "Mixed-traffic data collection started\n"
              << "Scenarios: " << totalScenarios << "\n"
              << "Each scenario has nSta mixed flows\n\n";

    for (int nSta : STA_VALUES)
    for (int ru   : RU_VALUES)
    for (int mcs  : MCS_VALUES)
    {
        scenarioDone++;

        // Run mixed simulation for 15 seconds
        auto flowResults = RunMixedScenario(
            nSta, ru, mcs, 15.0);

        // Write one row per flow result
        for (auto& r : flowResults)
        {
            csv << r.trafficType      << ","
                << r.pktSize          << ","
                << r.intervalMs       << ","
                << r.numStations      << ","
                << r.ru               << ","
                << r.mcs              << ","
                << std::fixed
                << std::setprecision(6)
                << r.throughput_mbps  << ","
                << r.mean_delay_ms    << ","
                << r.mean_jitter_ms   << ","
                << r.packet_loss_rate << ","
                << r.priority         << ","
                << r.twtMs            << "\n";
            rowsWritten++;
        }

        // Progress update
        if (scenarioDone % 10 == 0 || scenarioDone == totalScenarios)
        {
            std::cout << "  Scenario " << scenarioDone
                      << "/" << totalScenarios
                      << "  nSta=" << nSta
                      << "  RU="   << ru
                      << "  MCS="  << mcs
                      << "  rows so far=" << rowsWritten
                      << "\n";
        }
    }

    csv.close();

    std::cout << "\n✅  Done!\n"
              << "    Scenarios run : " << scenarioDone << "\n"
              << "    CSV rows      : " << rowsWritten  << "\n"
              << "    Output file   : " << outFile      << "\n"
              << "\nNext step: python3 2_train_model.py\n\n";
    return 0;
}
