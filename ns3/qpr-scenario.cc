// ns-3 scenario: simulates one sample file
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <fstream>
#include <map>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("QprScenario");

/* ------------------------------------------------------------------ traffic source */
class QprSource : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("QprSource").SetParent<Application>().AddConstructor<QprSource>();
        return tid;
    }

    void Configure(Address peer, uint8_t prio, double rate, bool onoff, double onMean, double offMean,
                   std::vector<uint32_t> sizes, std::vector<double> probs)
    {
        m_peer = peer;
        m_prio = prio;
        m_onoff = onoff;
        // keep mean rate under ON-OFF
        m_rate = onoff ? rate * (onMean + offMean) / onMean : rate;
        m_onMean = onMean;
        m_offMean = offMean;
        m_sizes = sizes;
        double c = 0;
        for (double p : probs)
        {
            c += p;
            m_cdf.push_back(c);
        }
    }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        m_socket->SetPriority(m_prio);   // classified by PrioQueueDisc via its priomap
        m_gap = CreateObject<ExponentialRandomVariable>();
        m_uni = CreateObject<UniformRandomVariable>();
        m_per = CreateObject<ExponentialRandomVariable>();
        if (m_onoff)
        {
            bool on = m_uni->GetValue() < m_onMean / (m_onMean + m_offMean);   // stationary start
            if (on)
            {
                StartOn();
            }
            else
            {
                m_event = Simulator::Schedule(Seconds(m_per->GetValue(m_offMean, 0)), &QprSource::StartOn, this);
            }
        }
        else
        {
            m_onEnd = Time::Max();
            ScheduleNext();
        }
    }

    void StopApplication() override
    {
        Simulator::Cancel(m_event);
        if (m_socket)
        {
            m_socket->Close();
        }
    }

    void StartOn()
    {
        m_onEnd = Simulator::Now() + Seconds(m_per->GetValue(m_onMean, 0));
        ScheduleNext();
    }

    void ScheduleNext()
    {
        Time next = Simulator::Now() + Seconds(m_gap->GetValue(1.0 / m_rate, 0));
        if (next > m_onEnd)  // ON period over: switch to OFF
        {
            m_event = Simulator::Schedule(m_onEnd - Simulator::Now() + Seconds(m_per->GetValue(m_offMean, 0)),
                                          &QprSource::StartOn, this);
            return;
        }
        m_event = Simulator::Schedule(next - Simulator::Now(), &QprSource::Send, this);
    }

    void Send()
    {
        double u = m_uni->GetValue();
        size_t k = 0;
        while (k + 1 < m_cdf.size() && u > m_cdf[k])
        {
            k++;
        }
        uint32_t payload = m_sizes[k] - 28;       // IP (20) + UDP (8) headers
        m_socket->Send(Create<Packet>(payload));
        ScheduleNext();
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    uint8_t m_prio{0};
    bool m_onoff{false};
    double m_rate{1}, m_onMean{1}, m_offMean{1};
    std::vector<uint32_t> m_sizes;
    std::vector<double> m_cdf;
    Ptr<ExponentialRandomVariable> m_gap, m_per;
    Ptr<UniformRandomVariable> m_uni;
    Time m_onEnd;
    EventId m_event;
};

/* ------------------------------------------------------------------ input */
struct FlowSpec
{
    uint32_t id, src, dst, tos;
    double rate;
    bool onoff;
    double on, off;
    std::vector<uint32_t> sizes;
    std::vector<double> probs;
};

int
main(int argc, char* argv[])
{
    std::string in, out;
    CommandLine cmd(__FILE__);
    cmd.AddValue("in", "sample file (scripts/make_ns3_inputs.py)", in);
    cmd.AddValue("out", "per-flow result file", out);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(in.empty() || out.empty(), "need --in and --out");

    std::ifstream f(in);
    NS_ABORT_MSG_IF(!f, "cannot open " << in);
    std::string line, key;
    std::getline(f, line);                        // header comment
    uint32_t N, L;
    f >> key >> N >> key >> L;
    std::vector<std::array<uint64_t, 3>> links(L);
    for (auto& l : links)
    {
        f >> l[0] >> l[1] >> l[2];
    }
    std::vector<std::array<uint32_t, 6>> nodecfg(N);   // policy, nq, bufferBytes, q(tos0..2)
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t n;
        f >> key >> n;
        for (auto& v : nodecfg[n])
        {
            f >> v;
        }
    }
    uint32_t R;
    f >> key >> R;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> nexthop;
    for (uint32_t i = 0; i < R; i++)
    {
        uint32_t v, d, h;
        f >> v >> d >> h;
        nexthop[{v, d}] = h;
    }
    uint32_t F;
    f >> key >> F;
    std::vector<FlowSpec> flows(F);
    for (auto& fl : flows)
    {
        uint32_t onoff, k;
        f >> fl.id >> fl.src >> fl.dst >> fl.tos >> fl.rate >> onoff >> fl.on >> fl.off >> k;
        fl.onoff = onoff;
        fl.sizes.resize(k);
        fl.probs.resize(k);
        for (uint32_t j = 0; j < k; j++)
        {
            f >> fl.sizes[j] >> fl.probs[j];
        }
    }
    double T, warm;
    uint32_t seed;
    f >> key >> T >> warm >> seed;
    NS_ABORT_MSG_IF(!f, "malformed sample file " << in);
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    /* topology */
    NodeContainer nodes;
    nodes.Create(N);
    InternetStackHelper stack;
    Ipv4StaticRoutingHelper staticRouting;
    stack.SetRoutingHelper(staticRouting);
    stack.Install(nodes);

    PointToPointHelper p2p;
    p2p.SetChannelAttribute("Delay", TimeValue(Seconds(0)));
    p2p.SetDeviceAttribute("Mtu", UintegerValue(1800));
    p2p.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("1p"));

    Ipv4AddressHelper addr;
    addr.SetBase("10.0.0.0", "255.255.255.252");
    // interface towards v, and v's address
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, Ipv4Address>> ifOf;
    std::vector<Ipv4Address> home(N);
    std::vector<bool> hasHome(N, false);
    for (auto& l : links)
    {
        uint32_t u = l[0], v = l[1];
        p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(l[2])));
        NetDeviceContainer d = p2p.Install(nodes.Get(u), nodes.Get(v));
        for (uint32_t side = 0; side < 2; side++)  // queue disc on each sending side
        {
            uint32_t n = side == 0 ? u : v;
            auto& c = nodecfg[n];
            QueueSizeValue qs(QueueSize(QueueSizeUnit::BYTES, c[2]));
            TrafficControlHelper tch;
            if (c[0] == 0)
            {
                tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", qs);
            }
            else
            {
                // map priority (ToS) to band
                std::ostringstream pm;
                for (uint32_t p = 0; p < 16; p++)
                {
                    pm << (p < 3 ? c[3 + p] : c[1] - 1) << (p < 15 ? " " : "");
                }
                uint16_t h = tch.SetRootQueueDisc("ns3::PrioQueueDisc", "Priomap", StringValue(pm.str()));
                auto cls = tch.AddQueueDiscClasses(h, c[1], "ns3::QueueDiscClass");
                for (uint32_t b = 0; b < c[1]; b++)
                {
                    tch.AddChildQueueDisc(h, cls[b], "ns3::FifoQueueDisc", "MaxSize", qs);
                }
            }
            tch.Install(d.Get(side));
        }
        Ipv4InterfaceContainer ic = addr.Assign(d);
        addr.NewNetwork();
        uint32_t iu = nodes.Get(u)->GetObject<Ipv4>()->GetInterfaceForDevice(d.Get(0));
        uint32_t iv = nodes.Get(v)->GetObject<Ipv4>()->GetInterfaceForDevice(d.Get(1));
        ifOf[{u, v}] = {iu, ic.GetAddress(1)};
        ifOf[{v, u}] = {iv, ic.GetAddress(0)};
        if (!hasHome[u]) { home[u] = ic.GetAddress(0); hasHome[u] = true; }
        if (!hasHome[v]) { home[v] = ic.GetAddress(1); hasHome[v] = true; }
    }
    for (auto& [vd, h] : nexthop)
    {
        auto [v, d] = vd;
        auto& [ifi, gw] = ifOf.at({v, h});
        staticRouting.GetStaticRouting(nodes.Get(v)->GetObject<Ipv4>())->AddHostRouteTo(home[d], gw, ifi);
    }

    /* traffic */
    for (auto& fl : flows)
    {
        uint16_t port = 10000 + fl.id;
        PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        sink.Install(nodes.Get(fl.dst)).Start(Seconds(0));
        Ptr<QprSource> app = CreateObject<QprSource>();
        app->Configure(InetSocketAddress(home[fl.dst], port), fl.tos, fl.rate, fl.onoff, fl.on, fl.off,
                       fl.sizes, fl.probs);
        nodes.Get(fl.src)->AddApplication(app);
        app->SetStartTime(Seconds(0));
        app->SetStopTime(Seconds(T));
    }

    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> mon = fmh.InstallAll();
    mon->Start(Seconds(warm));
    mon->Stop(Seconds(T + 1.0));  // let packets in flight arrive
    Simulator::Stop(Seconds(T + 1.5));
    Simulator::Run();
    mon->CheckForLostPackets();

    auto cls = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    std::vector<FlowMonitor::FlowStats> res(F);
    std::vector<bool> seen(F, false);
    for (auto& [fid, st] : mon->GetFlowStats())
    {
        auto t = cls->FindFlow(fid);
        if (t.protocol != 17 || t.destinationPort < 10000)
        {
            continue;
        }
        uint32_t id = t.destinationPort - 10000;
        res[id] = st;
        seen[id] = true;
    }
    std::ofstream o(out);
    o << "# window " << T - warm << "\n";
    o.precision(9);
    for (uint32_t i = 0; i < F; i++)
    {
        auto& s = res[i];
        o << i << " " << (seen[i] ? s.txPackets : 0) << " " << (seen[i] ? s.txBytes : 0) << " "
          << (seen[i] ? s.rxPackets : 0) << " " << (seen[i] ? s.rxBytes : 0) << " "
          << (seen[i] ? s.delaySum.GetSeconds() : 0) << " " << (seen[i] ? s.jitterSum.GetSeconds() : 0)
          << " " << (seen[i] ? s.lostPackets : 0) << "\n";
    }
    Simulator::Destroy();
    return 0;
}
