"use client";

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface FormSectionProps {
  title: string;
  children: ReactNode;
  delay?: number;
}

export function FormSection({ title, children, delay = 0 }: FormSectionProps) {
  return (
    <motion.div
      initial={{ x: 30, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      className="space-y-4"
    >
      <motion.h3 
        className="text-lg font-semibold bg-gradient-to-r from-blue-700 to-cyan-600 bg-clip-text text-transparent border-b border-blue-200 pb-2"
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.2 }}
      >
        {title}
      </motion.h3>
      <div className="space-y-4">
        {children}
      </div>
    </motion.div>
  );
}