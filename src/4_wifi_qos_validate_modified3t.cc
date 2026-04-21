/* =============================================================
 * 4_wifi_qos_validate.cc
 *
 * PURPOSE:
 *   Validate the ns-3-trained ML model by running it inside
 *   a Wi-Fi 6 (802.11ax) network topology and comparing
 *   QoS metrics against a uniform baseline.
 *
 * TWO MODES (--useML flag):
 *   --useML=false  →  Baseline: all stations get equal
 *                     RU, MCS, TWT, priority
 *   --useML=true   →  ML-driven: 3_predict.py is called
 *                     per flow; each station gets
 *                     differentiated QoS parameters
 *
 * TOPOLOGY:
 *   - 1 Wi-Fi 6 Access Point (802.11ax, 5 GHz, 80 MHz)
 *   - nSta stations (default 8) in a 2 m grid
 *   - Mixed traffic: VoIP / Video / HTTP / VPN / IoT
 *   - All stations within 5 m of AP (reliable links)
 *
 * OUTPUT FILES:
 *   results_<tag>.csv   — per-flow QoS parameters
 *   stats_<tag>.csv     — FlowMonitor measurements
 *
 * BUILD & RUN:
 *   cp 4_wifi_qos_validate.cc ~/ns-3-dev/scratch/
 *   cp 3_predict.py model.pkl feature_cols.pkl \
 *      label_encoder.pkl ~/ns-3-dev/
 *   cd ~/ns-3-dev && ./ns3 build
 *   ./ns3 run "scratch/4_wifi_qos_validate --useML=false"
 *   ./ns3 run "scratch/4_wifi_qos_validate --useML=true"
 * ============================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/applications-module.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WifiQoSValidate");

// ─────────────────────────────────────────────────────────────
// Association trace
// ─────────────────────────────────────────────────────────────
static uint32_t g_associated = 0;
void StationAssociated(std::string, Mac48Address)
{
    g_associated++;
}

// ─────────────────────────────────────────────────────────────
// Call 3_predict.py — returns {priority, ru, twt, mcs}
// ─────────────────────────────────────────────────────────────
std::vector<int> GetMLOutputs(int pktSize,
                               int intervalMs,
                               int nSta)
{
    std::string cmd =
        "python3 ./3_predict.py "
        + std::to_string(pktSize)   + " "
        + std::to_string(intervalMs) + " "
        + std::to_string(nSta);

    std::array<char, 256> buf;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {1, 10, 50, 6};
    while (fgets(buf.data(), buf.size(), pipe))
        result += buf.data();
    pclose(pipe);

    std::istringstream iss(result);
    int p, ru, twt, mcs;
    if (!(iss >> p >> ru >> twt >> mcs))
        return {1, 10, 50, 6};
    return {p, ru, twt, mcs};
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
double GetBandwidthMHz(int ru)
{
    if (ru >= 37) return 80.0;
    if (ru >= 18) return 40.0;
    if (ru >= 9)  return 20.0;
    if (ru >= 4)  return 10.0;
    return 5.0;
}

// RU + MCS → effective station data rate (Mbps)
double RuMcsToRateMbps(int ru, int mcs)
{
    double mcsRates[] = {
        8.6, 17.2, 25.8, 34.4,  51.6,
       68.8, 77.4, 86.0,103.2, 114.7, 129.0, 143.4
    };
    mcs = std::max(0, std::min(mcs, 11));
    double r = ((double)ru / 37.0) * mcsRates[mcs];
    return std::max(0.5, std::floor(r * 2.0) / 2.0);
}

std::string GetTrafficType(uint32_t i)
{
    // Same cycling order as 1_data_collector
    switch (i % 5) {
        case 0: return "VoIP";
        case 1: return "Video";
        case 2: return "HTTP";
        case 3: return "VPN";
        default:return "IoT";
    }
}

std::string GetPriorityName(int p)
{
    if (p >= 3) return "High";
    if (p >= 2) return "High";
    if (p >= 1) return "Medium";
    return "Low";
}

// ─────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    bool     useML   = true;
    uint32_t nSta    = 8;
    double   simTime = 60.0;

    CommandLine cmd;
    cmd.AddValue("useML",  "ML-driven QoS (true/false)", useML);
    cmd.AddValue("nSta",   "Number of stations",          nSta);
    cmd.AddValue("simTime","Simulation duration (s)",     simTime);
    cmd.Parse(argc, argv);

    std::string tag      = useML ? "with_ml" : "without_ml";
    std::string resCsv   = "results_" + tag + ".csv";
    std::string statsCsv = "stats_"   + tag + ".csv";

    std::cout << "\n========================================\n"
              << " Wi-Fi 6 QoS Validation  [" << tag << "]\n"
              << " nSta=" << nSta
              << "  simTime=" << simTime << "s\n"
              << "========================================\n\n";

    // ── AMPDU config ──────────────────────────────────────────
    Config::SetDefault("ns3::WifiMac::BE_MaxAmpduSize",
                       UintegerValue(65535));
    Config::SetDefault("ns3::WifiMac::VI_MaxAmpduSize",
                       UintegerValue(65535));
    Config::SetDefault("ns3::WifiMac::VO_MaxAmpduSize",
                       UintegerValue(65535));

    // ── Collect per-flow QoS params before building network ───
    // (ML calls happen here so we know MCS before install)
    struct FlowCfg {
        uint32_t    pktSize;
        int         intervalMs;
        int         priority;
        int         ru;
        int         twt;
        int         mcs;
        double      rateMbps;
        double      bwMHz;
        int         effIntMs;
        double      startTime;
        std::string trafficType;
        std::string priorityName;
    };

    std::vector<FlowCfg> cfg(nSta);
    int totalRuDemand = 0; // NEW: Track total requested RUs

    // ==========================================================
    // PASS 1: Generate ML predictions and calculate total demand
    // ==========================================================
    for (uint32_t i = 0; i < nSta; i++)
    {
        auto& c = cfg[i];

        // Traffic pattern — same as data collector
        switch (i % 5) {
            case 0: c.pktSize=160;  c.intervalMs=20;   break; // VoIP
            case 1: c.pktSize=1400; c.intervalMs=40;   break; // Video
            case 2: c.pktSize=800;  c.intervalMs=100;  break; // HTTP
            case 3: c.pktSize=900;  c.intervalMs=80;   break; // VPN
            default:c.pktSize=64;   c.intervalMs=1000; break; // IoT
        }

        if (useML) {
            // 1. Fetch predictions
            auto out = GetMLOutputs(c.pktSize, c.intervalMs, (int)nSta);
            
            // 2. Keep ONLY the ML-predicted Priority and TWT
            c.priority = std::max(0, std::min(out[0],  3));
            c.twt      = std::max(5, std::min(out[2],500));
            
            // 3. Strict mathematical assignment based on Priority
            switch (c.priority) {
                case 3: c.ru = 1;  c.mcs = 4; break; // VoIP
                case 2: c.ru = 18; c.mcs = 9; break; // Video
                case 1: c.ru = 9;  c.mcs = 6; break; // Best Effort
                case 0: 
                default:c.ru = 1;  c.mcs = 0; break; // IoT
            }
        } else {
            // Baseline: every station gets equal share
            c.priority = 1;
            c.ru       = std::max(1, (int)(37 / (int)nSta));
            c.twt      = 50;
            c.mcs      = 6;
        }

        // Add this station's required RU to our global tracker
        totalRuDemand += c.ru; 
    }

 // ==========================================================
    // PASS 2: Heuristic AI/AD Scheduler & Fairness Allocation
    // Inspired by HOBO-UORA (Rehman et al.) & CR-OFDMA (Li et al.)
    // ==========================================================
    double ai_ad_multiplier = 1.0;
    
    if (useML) {
        if (totalRuDemand > 37) {
            // DENSE SCENARIO: Additive Decrease of Access (Increase TWT)
            // Prevent AP crash by forcing stations to sleep longer to avoid collisions.
            ai_ad_multiplier = (double)totalRuDemand / 37.0; 
            std::cout << "\n[DENSE NETWORK] Total Demand (" << totalRuDemand 
                      << " RUs) > Capacity. Applying HOBO-UORA TWT Penalty: " 
                      << std::fixed << std::setprecision(2) << ai_ad_multiplier << "x\n";
        } else if (totalRuDemand < 37) {
            // SPARSE SCENARIO: Additive Increase of Access (Decrease TWT)
            // Eliminate idle RUs by letting stations wake up much faster.
            ai_ad_multiplier = (double)totalRuDemand / 37.0; // Produces a multiplier < 1.0
            std::cout << "\n[SPARSE NETWORK] Total Demand (" << totalRuDemand 
                      << " RUs) < Capacity. Applying HOBO-UORA TWT Boost: " 
                      << std::fixed << std::setprecision(2) << ai_ad_multiplier << "x\n";
        }
    }

    for (uint32_t i = 0; i < nSta; i++) 
    {
        auto& c = cfg[i];

        if (useML) {
            // 1. FREQUENCY DOMAIN: QoS Fairness (CR-OFDMA Logic)
            // If the network is extremely dense, shrink high-bandwidth RUs 
            // to ensure lower priorities are not completely starved.
            if (ai_ad_multiplier > 2.0 && c.priority == 2) {
                c.ru = 9; // Fairness: Shrink Video from 40MHz to 20MHz
            }
            if (ai_ad_multiplier > 1.5 && c.priority == 1) {
                c.ru = 4; // Fairness: Shrink Best Effort from 20MHz to 10MHz
            }

            // 2. TIME DOMAIN: AI/AD Back-off (HOBO-UORA Logic)
            // Apply the AI/AD multiplier to all non-IoT traffic.
            if (c.priority >= 1) {
                // If dense (multiplier > 1.0), TWT increases (more delay, less collision).
                // If sparse (multiplier < 1.0), TWT decreases (less delay, zero idle RUs).
                // We enforce a hard floor of 2ms so the physics engine doesn't break.
                c.twt = std::max(2, (int)(c.twt * ai_ad_multiplier));
            }
        }

        // Finalize rates and bandwidths based on the validated/scaled RUs
        c.rateMbps = RuMcsToRateMbps(c.ru, c.mcs);
        c.bwMHz    = GetBandwidthMHz(c.ru);

        // Effective send interval — never faster than RU allows
        double ruMaxMbps = ((double)c.ru / 37.0) * 600.0;
        double minIntMs  = (c.pktSize * 8.0) / (ruMaxMbps * 1e6) * 1000.0;
        c.effIntMs = std::max(c.intervalMs, (int)std::ceil(minIntMs));

        // TWT controls when station starts transmitting
        c.startTime    = 2.0 + (double)c.twt / 1000.0 + i * 0.1;
        c.trafficType  = GetTrafficType(i);
        c.priorityName = GetPriorityName(c.priority);
    }

    // ── Nodes ─────────────────────────────────────────────────
    NodeContainer staNodes, apNode;
    staNodes.Create(nSta);
    apNode.Create(1);

    // ── Channel — clean indoor, no extra path loss ─────────────
    YansWifiChannelHelper channel =
        YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("ChannelSettings",
            StringValue("{0, 80, BAND_5GHZ, 0}"));

    // ── AP ────────────────────────────────────────────────────
    Ssid ssid("wifi6-qos");
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    WifiMacHelper macAp;
    macAp.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice =
        wifi.Install(phy, macAp, apNode);

    // ── ALL STAs installed together before IP assignment ───────
    WifiMacHelper macSta;
    macSta.SetType("ns3::StaWifiMac",
                   "Ssid",          SsidValue(ssid),
                   "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices =
        wifi.Install(phy, macSta, staNodes);

    // ── Mobility — neat grid, all within 5 m ──────────────────
    MobilityHelper mobility;
    mobility.SetPositionAllocator(
        "ns3::GridPositionAllocator",
        "MinX",      DoubleValue(1.0),
        "MinY",      DoubleValue(1.0),
        "DeltaX",    DoubleValue(2.0),
        "DeltaY",    DoubleValue(2.0),
        "GridWidth", UintegerValue(4));
    mobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");
    mobility.Install(staNodes);

    MobilityHelper apMob;
    apMob.SetPositionAllocator(
        "ns3::GridPositionAllocator",
        "MinX", DoubleValue(0.0),
        "MinY", DoubleValue(0.0));
    apMob.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");
    apMob.Install(apNode);

    // ── Internet stack + IPs (all at once) ────────────────────
    InternetStackHelper internet;
    internet.Install(apNode);
    internet.Install(staNodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apIface  = ipv4.Assign(apDevice);
    ipv4.Assign(staDevices);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ── Association trace ─────────────────────────────────────
    Config::Connect(
        "/NodeList/*/DeviceList/*"
        "/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/Assoc",
        MakeCallback(&StationAssociated));

    // ── FlowMonitor ───────────────────────────────────────────
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // ── Sink on AP ────────────────────────────────────────────
    uint16_t port = 9;
    PacketSinkHelper sinkH(
        "ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sinkH.Install(apNode.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(simTime));

    // ── Results CSV ───────────────────────────────────────────
    std::ofstream resCsvFile(resCsv);
    resCsvFile << "flow_id,traffic_type,priority_name,"
               << "priority,ru,twt_ms,mcs,bandwidth_mhz,"
               << "rate_mbps,packet_size,interval_ms,"
               << "effective_interval_ms,start_time_s\n";

    // ── Install one OnOff app per station ─────────────────────
    for (uint32_t i = 0; i < nSta; i++)
    {
        auto& c = cfg[i];

        // App send rate = pktSize * 8 / effInterval
        // --- STRESS TEST MODIFICATION ---
        // 1. We ignore the effIntMs throttle entirely.
        // 2. We multiply the raw application rate by 150 to flood the network.
        double stressMultiplier = 10.0; 
        double bps = ((double)c.pktSize * 8.0 / (double)c.intervalMs * 1000.0) * stressMultiplier;
        
        std::ostringstream rateStr;
        rateStr << (uint64_t)bps << "bps";

        std::cout << "Flow " << std::setw(2) << i
                  << " | " << std::setw(5) << c.trafficType
                  << " | " << std::setw(6) << c.priorityName
                  << " | RU="    << std::setw(2) << c.ru
                  << " | MCS="   << c.mcs
                  << " | Rate="  << std::setw(6) << c.rateMbps
                  << " Mbps"
                  << " | TWT="   << c.twt << "ms"
                  << " | BW="    << c.bwMHz << "MHz"
                  << " | Start=" << std::fixed
                  << std::setprecision(2) << c.startTime << "s\n";

        resCsvFile << i               << ","
                   << c.trafficType   << ","
                   << c.priorityName  << ","
                   << c.priority      << ","
                   << c.ru            << ","
                   << c.twt           << ","
                   << c.mcs           << ","
                   << c.bwMHz         << ","
                   << c.rateMbps      << ","
                   << c.pktSize       << ","
                   << c.intervalMs    << ","
                   << c.effIntMs      << ","
                   << c.startTime     << "\n";

        OnOffHelper onoff(
            "ns3::UdpSocketFactory",
            InetSocketAddress(apIface.GetAddress(0), port));
        onoff.SetConstantRate(DataRate(rateStr.str()),
                              c.pktSize);
        onoff.SetAttribute("OnTime",  StringValue(
            "ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue(
            "ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer app = onoff.Install(staNodes.Get(i));
        app.Start(Seconds(c.startTime));
        app.Stop(Seconds(simTime - 1.0));
    }

    resCsvFile.close();

    // ── Run ───────────────────────────────────────────────────
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    std::cout << "\nStations associated: "
              << g_associated << "/" << nSta << "\n";

    // ── FlowMonitor stats ─────────────────────────────────────
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(
            flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats =
        monitor->GetFlowStats();

    std::ofstream statsCsvFile(statsCsv);
    statsCsvFile << "flow_id,traffic_type,src_ip,dst_ip,"
                 << "tx_packets,rx_packets,"
                 << "tx_bytes,rx_bytes,"
                 << "throughput_mbps,mean_delay_ms,"
                 << "mean_jitter_ms,packet_loss_rate\n";

    std::cout << "\n--- Flow Statistics [" << tag << "] ---\n";
    std::cout << std::left
              << std::setw(8)  << "FlowID"
              << std::setw(8)  << "Type"
              << std::setw(14) << "Tput(Mbps)"
              << std::setw(12) << "Delay(ms)"
              << std::setw(12) << "Jitter(ms)"
              << "Loss(%)\n"
              << std::string(60, '-') << "\n";

    // Per-type accumulators
    std::map<std::string,double> tTput, tDelay,
                                  tJitter, tLoss;
    std::map<std::string,int>    tCount;
    double totT=0, totD=0, totJ=0, totL=0;
    int fc = 0;

    // Map FlowId → traffic type
    // FlowMonitor flow IDs start at 1, match app install order
    std::vector<std::string> flowTypeMap;
    for (uint32_t i = 0; i < nSta; i++)
        flowTypeMap.push_back(cfg[i].trafficType);

    for (auto& kv : stats)
    {
        auto& s = kv.second;
        Ipv4FlowClassifier::FiveTuple t =
            classifier->FindFlow(kv.first);

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

        std::string ttype = "Unknown";
        uint32_t idx = kv.first - 1;
        if (idx < flowTypeMap.size())
            ttype = flowTypeMap[idx];

        std::cout << std::left
                  << std::setw(8)  << kv.first
                  << std::setw(8)  << ttype
                  << std::setw(14) << std::fixed
                  << std::setprecision(4) << tput
                  << std::setw(12) << delay
                  << std::setw(12) << jitter
                  << loss * 100 << "\n";

        statsCsvFile << kv.first             << ","
                     << ttype                << ","
                     << t.sourceAddress      << ","
                     << t.destinationAddress << ","
                     << s.txPackets          << ","
                     << s.rxPackets          << ","
                     << s.txBytes            << ","
                     << s.rxBytes            << ","
                     << tput    << ","
                     << delay   << ","
                     << jitter  << ","
                     << loss    << "\n";

        tTput[ttype]   += tput;
        tDelay[ttype]  += delay;
        tJitter[ttype] += jitter;
        tLoss[ttype]   += loss;
        tCount[ttype]++;

        totT+=tput; totD+=delay;
        totJ+=jitter; totL+=loss;
        fc++;
    }

    statsCsvFile.close();

    // ── Per traffic type summary ───────────────────────────────
    if (fc > 0)
    {
        std::cout << "\n--- Per Traffic Type ["
                  << tag << "] ---\n";
        std::cout << std::left
                  << std::setw(8)  << "Type"
                  << std::setw(14) << "Tput(Mbps)"
                  << std::setw(12) << "Delay(ms)"
                  << std::setw(12) << "Jitter(ms)"
                  << "Loss(%)\n"
                  << std::string(56, '-') << "\n";

        for (auto& tp :
             std::vector<std::string>
             {"VoIP","Video","HTTP","VPN","IoT"})
        {
            if (!tCount.count(tp) || tCount[tp]==0)
                continue;
            int c = tCount[tp];
            std::cout << std::left
                      << std::setw(8)  << tp
                      << std::setw(14) << std::fixed
                      << std::setprecision(4)
                      << tTput[tp]/c
                      << std::setw(12) << tDelay[tp]/c
                      << std::setw(12) << tJitter[tp]/c
                      << tLoss[tp]/c*100 << "\n";
        }

        std::cout << "\n=== OVERALL SUMMARY ["
                  << tag << "] ===\n"
                  << "Flows         : " << fc << "\n"
                  << "Avg Throughput: "
                  << std::fixed << std::setprecision(4)
                  << totT/fc << " Mbps\n"
                  << "Avg Delay     : "
                  << totD/fc << " ms\n"
                  << "Avg Jitter    : "
                  << totJ/fc << " ms\n"
                  << "Avg Loss      : "
                  << totL/fc*100 << "%\n";
    }

    Simulator::Destroy();

    std::cout << "\n✅ " << resCsv
              << "\n✅ " << statsCsv << "\n\n";
    return 0;
}
