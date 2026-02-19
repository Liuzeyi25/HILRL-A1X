#!/bin/bash
# 诊断到 172.17.0.15 的连接问题

TARGET_IP="172.17.0.15"

echo "=========================================="
echo "🔍 诊断到 ${TARGET_IP} 的网络连接"
echo "=========================================="
echo ""

# 1. 检查本机 IP
echo "1️⃣ 本机 IP 地址:"
ip addr show | grep "inet " | awk '{print "   ", $2, $NF}'
echo ""

# 2. 检查路由表
echo "2️⃣ 路由表:"
ip route | head -10 | while read line; do echo "    $line"; done
echo ""

# 3. Ping 测试
echo "3️⃣ Ping 测试:"
if ping -c 2 -W 2 ${TARGET_IP} &>/dev/null; then
    echo "   ✅ Ping 成功"
    ping -c 2 ${TARGET_IP} | grep "time=" | head -2 | while read line; do echo "    $line"; done
else
    echo "   ❌ Ping 失败 (可能防火墙阻止 ICMP)"
fi
echo ""

# 4. 检查是否可以到达该网段
echo "4️⃣ 网络可达性:"
if ip route get ${TARGET_IP} &>/dev/null; then
    echo "   ✅ 路由存在"
    ip route get ${TARGET_IP} | while read line; do echo "    $line"; done
else
    echo "   ❌ 无路由到 ${TARGET_IP}"
fi
echo ""

# 5. 端口扫描
echo "5️⃣ 端口连接测试:"
for port in 3333 3334; do
    echo "   测试端口 ${port}..."
    if timeout 3 bash -c "echo >/dev/tcp/${TARGET_IP}/${port}" 2>/dev/null; then
        echo "   ✅ 端口 ${port} 可连接"
    else
        result=$(nc -zv -w 2 ${TARGET_IP} ${port} 2>&1)
        if echo "$result" | grep -q "succeeded"; then
            echo "   ✅ 端口 ${port} 可连接"
        elif echo "$result" | grep -q "refused"; then
            echo "   ⚠️  端口 ${port} 连接被拒绝（服务未启动）"
        elif echo "$result" | grep -q "No route"; then
            echo "   ❌ 端口 ${port} 无路由到主机"
        else
            echo "   ❌ 端口 ${port} 无法连接: $result"
        fi
    fi
done
echo ""

# 6. ARP 表检查
echo "6️⃣ ARP 表（检查是否有该 IP 的 MAC 地址）:"
arp -n | grep "${TARGET_IP}" || echo "   ⚠️  ARP 表中无此 IP"
echo ""

# 7. 防火墙检查
echo "7️⃣ 防火墙状态:"
if command -v ufw &>/dev/null; then
    sudo ufw status 2>/dev/null | head -5 | while read line; do echo "    $line"; done
elif command -v firewall-cmd &>/dev/null; then
    sudo firewall-cmd --state 2>/dev/null
else
    echo "   ℹ️  未检测到 ufw 或 firewall-cmd"
fi
echo ""

# 8. 建议
echo "=========================================="
echo "💡 排查建议："
echo "=========================================="
echo ""

# 检查是否在同一网段
LOCAL_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -1)
LOCAL_SUBNET=$(echo $LOCAL_IP | cut -d. -f1-3)
TARGET_SUBNET=$(echo $TARGET_IP | cut -d. -f1-3)

if [ "$LOCAL_SUBNET" = "$TARGET_SUBNET" ]; then
    echo "✅ 本机和目标在同一子网 ($LOCAL_SUBNET.0/24)"
    echo ""
    echo "可能的问题："
    echo "1. 目标服务器的防火墙阻止了连接"
    echo "2. 目标服务器的服务未启动（端口未监听）"
    echo "3. 网络交换机或路由器配置问题"
    echo ""
    echo "建议操作："
    echo "• 在 ${TARGET_IP} 上运行: sudo ufw allow 3333/tcp"
    echo "• 在 ${TARGET_IP} 上运行: sudo ufw allow 3334/tcp"
    echo "• 在 ${TARGET_IP} 上检查服务是否启动: ss -tlnp | grep -E '3333|3334'"
else
    echo "⚠️  本机 ($LOCAL_SUBNET.x) 和目标 ($TARGET_SUBNET.x) 不在同一子网"
    echo ""
    echo "可能的问题："
    echo "1. 需要通过路由器/网关转发"
    echo "2. 不同的 Docker 网络或 VLAN"
    echo "3. VPN 或隧道配置问题"
    echo ""
    echo "建议操作："
    echo "• 检查网关配置: ip route show default"
    echo "• 如果都在 Docker 内，使用 Docker 网络桥接"
    echo "• 如果跨网段，确保路由器允许流量通过"
fi
echo ""
