"use client";

import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ReactNode } from 'react';

interface AnimatedCardProps {
  children: ReactNode;
  title?: string;
  description?: string;
  delay?: number;
  className?: string;
}

export function AnimatedCard({ children, title, description, delay = 0, className = "" }: AnimatedCardProps) {
  return (
    <motion.div
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay, duration: 0.6, ease: "easeOut" }}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className={className}
    >
      <Card className="bg-white/80 backdrop-blur-sm border-blue-200/50 shadow-lg hover:shadow-xl transition-all duration-300">
        {(title || description) && (
          <CardHeader>
            {title && (
              <CardTitle className="bg-gradient-to-r from-blue-700 to-cyan-600 bg-clip-text text-transparent">
                {title}
              </CardTitle>
            )}
            {description && (
              <CardDescription className="text-blue-600">
                {description}
              </CardDescription>
            )}
          </CardHeader>
        )}
        <CardContent>
          {children}
        </CardContent>
      </Card>
    </motion.div>
  );
}