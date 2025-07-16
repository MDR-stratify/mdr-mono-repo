"use client";

import { motion } from 'framer-motion';
import { Activity, Shield, Zap, TrendingUp, Users, Globe } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

const stats = [
  { icon: Shield, label: "Accuracy", value: "80%" },
  { icon: Zap, label: "Speed", value: "<2s" },
  { icon: Users, label: "Data points", value: "900K+" },
  { icon: Globe, label: "Countries", value: "150+" }
];

const features = [
  {
    icon: Activity,
    title: "Real-time Analysis",
    description: "Instant MDR risk assessment using advanced ML algorithms"
  },
  // {
  //   icon: TrendingUp,
  //   title: "Predictive Insights",
  //   description: "Evidence-based predictions to optimize antibiotic selection"
  // },
  // {
  //   icon: Shield,
  //   title: "Clinical Validation",
  //   description: "Validated across multiple healthcare settings in LMICs"
  // }
];

export function Sidebar() {
  return (
    <motion.div
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="w-80 bg-gradient-to-b from-blue-50 via-cyan-50 to-blue-100 border-r border-blue-200 p-6 overflow-y-auto"
    >
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="text-center"
        >
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-cyan-600 bg-clip-text text-transparent">
            MDR Stratify
          </h2>
          <p className="text-sm text-blue-600 mt-1">AI-Powered MDR Prediction</p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="grid grid-cols-2 gap-3"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5 + index * 0.1, duration: 0.3 }}
              whileHover={{ scale: 1.05 }}
              className="bg-white/70 backdrop-blur-sm rounded-lg p-3 text-center border border-blue-200/50 shadow-sm"
            >
              <stat.icon className="w-5 h-5 text-blue-600 mx-auto mb-1" />
              <div className="text-lg font-bold text-blue-800">{stat.value}</div>
              <div className="text-xs text-blue-600">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="space-y-4"
        >
          <h3 className="text-lg font-semibold text-blue-800 mb-3">Key Features</h3>
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.7 + index * 0.1, duration: 0.4 }}
              whileHover={{ x: 5 }}
              className="bg-white/60 backdrop-blur-sm rounded-lg p-4 border border-blue-200/50 shadow-sm"
            >
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center flex-shrink-0">
                  <feature.icon className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h4 className="font-medium text-blue-800 text-sm">{feature.title}</h4>
                  <p className="text-xs text-blue-600 mt-1 leading-relaxed">{feature.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* About MDR */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          <Card className="bg-gradient-to-br from-blue-100/80 to-cyan-100/80 border-blue-200/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <h3 className="text-sm font-semibold text-blue-800 mb-2">About MDR</h3>
              <p className="text-xs text-blue-700 leading-relaxed">
                Multi-Drug Resistance (MDR) occurs when pathogens are resistant to two or more antibiotics. Early prediction helps optimize treatment in resource-limited settings.
              </p>
            </CardContent>
          </Card>
        </motion.div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.5 }}
          className="text-center pt-4 border-t border-blue-200/50"
        >
          <p className="text-xs text-blue-500">
            Optimizing antibiotic use in LMICs
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
}