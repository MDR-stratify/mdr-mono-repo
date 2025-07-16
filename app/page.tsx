"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  AlertCircle,
  CheckCircle,
  XCircle,
  Activity,
  TrendingUp,
  Users,
} from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { AnimatedCard } from "@/components/AnimatedCard";
import { AnimatedButton } from "@/components/AnimatedButton";
import { FormSection } from "@/components/FormSection";
import { LoadingSpinner } from "@/components/LoadingSpinner";

const PATHOGENS = ["Staphylococcus aureus"];
const PHENOTYPES = ["MSSA", "MRSA"];
const COUNTRIES = [
  "Europe",
  "Asia",
  "North America",
  "South America",
  "Middle East",
  "Oceania",
  "Africa",
];
const SEXES = ["Male", "Female"];
const AGE_GROUPS = [
  "0-2 Years",
  "3-12 Years",
  "13-18 Years",
  "19-64 Years",
  "65-84 Years",
  "85+ Years",
  "Unknown",
];
const WARDS = [
  "Medicine General",
  "Surgery General",
  "ICU",
  "Emergency Room",
  "Pediatric General",
  "Clinic / Office",
  "Pediatric ICU",
  "Nursing Home / Rehab",
  "Unknown",
];
const SPECIMEN_TYPES = [
  "Wound",
  "Blood",
  "Sputum",
  "Abscess",
  "Endotracheal aspirate",
  "Gastric Abscess",
  "Skin: Other",
  "Ulcer",
  "Urine",
  "Bronchus",
  "Bronchoalveolar lavage",
  "Skin",
  "Trachea",
  "Cellulitis",
  "Peritoneal Fluid",
  "Respiratory: Other",
  "Decubitus",
  "Burn",
  "Nose",
  "Furuncle",
  "Catheters",
  "Exudate",
  "Impetiginous lesions",
  "Tissue Fluid",
  "Thoracentesis Fluid",
  "Abdominal Fluid",
  "Ear",
  "Intestinal: Other",
  "Eye",
  "Bone",
  "Synovial Fluid",
  "Lungs",
  "Throat",
  "None Given",
  "Bodily Fluids",
  "Carbuncle",
  "Aspirate",
  "HEENT: Other",
  "Pleural Fluid",
  "Respiratory: Sinuses",
  "Muscle",
  "Bladder",
  "Genitourinary: Other",
  "Gall Bladder",
  "Vagina",
  "Stomach",
  "Drains",
  "Urethra",
  "CSF",
];
const IN_OUT_PATIENT = ["Inpatient", "None Given", "Outpatient", "Other"];

interface AntibioticResult {
  name: string;
  resistance: boolean;
  confidence: number;
}

interface PredictionResult {
  overall_mdr_risk: boolean;
  mdr_confidence: number;
  risk_level: string;
  resistant_antibiotics: AntibioticResult[];
  susceptible_antibiotics: AntibioticResult[];
  total_resistant_count: number;
  resistance_percentage: number;
}

export default function Home() {
  const [formData, setFormData] = useState({
    pathogen: "",
    phenotype: "",
    country: "",
    sex: "",
    age_group: "",
    ward: "",
    specimen_type: "",
    in_out_patient: "",
    year: new Date().getFullYear().toString(),
  });

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,
          year: parseInt(formData.year),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const data = await response.json();
      console.log(data);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-blue-100 flex">
      <div className="hidden md:flex">
        <Sidebar />
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto"
        >
          {/* Header */}
          <motion.div
            initial={{ y: -30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-center mb-8"
          >
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-700 via-cyan-600 to-blue-800 bg-clip-text text-transparent mb-4">
              MDR Stratify
            </h1>
            <p className="text-xl text-blue-600 max-w-2xl mx-auto leading-relaxed">
              AI-driven MDR Prediction for Optimized Antibiotic Use in LMICs
            </p>
          </motion.div>

          {/* Main Form */}
          <AnimatedCard
            title="Patient Assessment"
            description="Enter patient and pathogen data based on the trained model features"
            delay={0.3}
            className="mb-8"
          >
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Patient Demographics */}
              <FormSection title="Patient Demographics" delay={0.4}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="age_group"
                      className="text-blue-700 font-medium"
                    >
                      Age Group
                    </Label>
                    <Select
                      value={formData.age_group}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, age_group: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select age group" />
                      </SelectTrigger>
                      <SelectContent>
                        {AGE_GROUPS.map((group) => (
                          <SelectItem key={group} value={group}>
                            {group}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label htmlFor="sex" className="text-blue-700 font-medium">
                      Sex
                    </Label>
                    <Select
                      value={formData.sex}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, sex: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select sex" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                      </SelectContent>
                    </Select>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="country"
                      className="text-blue-700 font-medium"
                    >
                      Continent
                    </Label>
                     <Select
                      value={formData.country}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, country: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select continent" />
                      </SelectTrigger>
                      <SelectContent>
                        {COUNTRIES.map((country) => (
                          <SelectItem key={country} value={country}>
                            {country}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label htmlFor="year" className="text-blue-700 font-medium">
                      Year
                    </Label>
                    <Input
                      id="year"
                      type="number"
                      value={formData.year}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          year: e.target.value,
                        }))
                      }
                      className="border-blue-200 focus:border-blue-400 focus:ring-blue-400"
                      required
                    />
                  </motion.div>
                </div>
              </FormSection>

              {/* Clinical Context */}
              <FormSection title="Clinical Context" delay={0.5}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label htmlFor="ward" className="text-blue-700 font-medium">
                      Ward/Department
                    </Label>
                    <Select
                      value={formData.ward}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, ward: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select ward" />
                      </SelectTrigger>
                      <SelectContent>
                        {WARDS.map((ward) => (
                          <SelectItem key={ward} value={ward}>
                            {ward}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="specimen_type"
                      className="text-blue-700 font-medium"
                    >
                      Specimen Type
                    </Label>
                    <Select
                      value={formData.specimen_type}
                      onValueChange={(value) =>
                        setFormData((prev) => ({
                          ...prev,
                          specimen_type: value,
                        }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select specimen type" />
                      </SelectTrigger>
                      <SelectContent>
                        {SPECIMEN_TYPES.map((type) => (
                          <SelectItem key={type} value={type}>
                            {type}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="in_out_patient"
                      className="text-blue-700 font-medium"
                    >
                      Patient Type
                    </Label>
                    <Select
                      value={formData.in_out_patient}
                      onValueChange={(value) =>
                        setFormData((prev) => ({
                          ...prev,
                          in_out_patient: value,
                        }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select patient type" />
                      </SelectTrigger>
                      <SelectContent>
                        {IN_OUT_PATIENT.map((type) => (
                          <SelectItem key={type} value={type}>
                            {type}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>
                </div>
              </FormSection>

              {/* Pathogen Information */}
              <FormSection title="Pathogen Information" delay={0.6}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="pathogen"
                      className="text-blue-700 font-medium"
                    >
                      Pathogen
                    </Label>
                    <Select
                      value={formData.pathogen}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, pathogen: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select pathogen" />
                      </SelectTrigger>
                      <SelectContent>
                        {PATHOGENS.map((pathogen) => (
                          <SelectItem key={pathogen} value={pathogen}>
                            {pathogen}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Label
                      htmlFor="phenotype"
                      className="text-blue-700 font-medium"
                    >
                      Phenotype
                    </Label>
                    <Select
                      value={formData.phenotype}
                      onValueChange={(value) =>
                        setFormData((prev) => ({ ...prev, phenotype: value }))
                      }
                    >
                      <SelectTrigger className="border-blue-200 focus:border-blue-400">
                        <SelectValue placeholder="Select phenotype" />
                      </SelectTrigger>
                      <SelectContent>
                        {PHENOTYPES.map((phenotype) => (
                          <SelectItem key={phenotype} value={phenotype}>
                            {phenotype}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </motion.div>
                </div>
              </FormSection>

              {/* Submit Button */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7, duration: 0.5 }}
                className="flex justify-center pt-6"
              >
                <AnimatedButton
                  type="submit"
                  disabled={isLoading}
                  className="px-12 py-3 text-lg font-semibold"
                >
                  {isLoading ? (
                    <>
                      <LoadingSpinner />
                      Analyzing Patient Data...
                    </>
                  ) : (
                    <>
                      <Activity className="mr-2 h-5 w-5" />
                      Predict MDR Risk
                    </>
                  )}
                </AnimatedButton>
              </motion.div>
            </form>
          </AnimatedCard>

          {/* Error Alert */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ duration: 0.3 }}
              >
                <Alert
                  className="mb-6 border-red-200 bg-red-50/80 backdrop-blur-sm"
                  variant="destructive"
                >
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          { result &&  (<AnimatedCard delay={0}>
            <div className="space-y-6">
             
              
                <>
                  {/* Main Result */}
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2, duration: 0.4 }}
                    className={`p-6 rounded-xl border-2 ${
                      result.overall_mdr_risk
                        ? "bg-gradient-to-br from-red-50 to-orange-50 border-red-200"
                        : "bg-gradient-to-br from-green-50 to-emerald-50 border-green-200"
                    }`}
                  >
                    <div className="flex items-center justify-center mb-4">
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{
                          delay: 0.4,
                          duration: 0.5,
                          type: "spring",
                        }}
                        className={`w-16 h-16 rounded-full flex items-center justify-center ${
                          result.overall_mdr_risk
                            ? "bg-red-100"
                            : "bg-green-100"
                        }`}
                      >
                        {result.overall_mdr_risk ? (
                          <XCircle className="h-8 w-8 text-red-600" />
                        ) : (
                          <CheckCircle className="h-8 w-8 text-green-600" />
                        )}
                      </motion.div>
                    </div>

                    <motion.h3
                      initial={{ y: 10, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.5, duration: 0.4 }}
                      className={`text-2xl font-bold text-center mb-2 ${
                        result.overall_mdr_risk
                          ? "text-red-700"
                          : "text-green-700"
                      }`}
                    >
                      {result.overall_mdr_risk
                        ? "High MDR Risk Detected"
                        : "Low MDR Risk"}
                    </motion.h3>

                    <motion.p
                      initial={{ y: 10, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.6, duration: 0.4 }}
                      className={`text-center ${
                        result.overall_mdr_risk
                          ? "text-red-600"
                          : "text-green-600"
                      }`}
                    >
                      The patient is{" "}
                      {result.overall_mdr_risk ? "likely" : "unlikely"} to have
                      Multi-Drug Resistance.
                    </motion.p>
                  </motion.div>

                  {/* Metrics + Resistance Info */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Confidence */}
                    <motion.div
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: 0.7, duration: 0.4 }}
                      className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-200"
                    >
                      <div className="flex items-center space-x-3">
                        <TrendingUp className="w-8 h-8 text-blue-600" />
                        <div>
                          <p className="text-sm text-blue-600 font-medium">
                            Confidence Level
                          </p>
                          <p className="text-2xl font-bold text-blue-800">
                            {(result.mdr_confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </motion.div>

                    {/* Risk Level */}
                    <motion.div
                      initial={{ y: 20, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.8, duration: 0.4 }}
                      className="bg-gradient-to-br from-purple-50 to-pink-50 p-4 rounded-lg border border-purple-200"
                    >
                      <div className="flex items-center space-x-3">
                        <Activity className="w-8 h-8 text-purple-600" />
                        <div>
                          <p className="text-sm text-purple-600 font-medium">
                            Risk Level
                          </p>
                          <Badge
                            variant={
                              result.risk_level === "High"
                                ? "destructive"
                                : "default"
                            }
                            className="text-lg px-3 py-1"
                          >
                            {result.risk_level}
                          </Badge>
                        </div>
                      </div>
                    </motion.div>

                    {/* Recommendation */}
                    <motion.div
                      initial={{ x: 20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: 0.9, duration: 0.4 }}
                      className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200"
                    >
                      <div className="flex items-center space-x-3">
                        <Users className="w-8 h-8 text-green-600" />
                        <div>
                          <p className="text-sm text-green-600 font-medium">
                            Recommendation
                          </p>
                          <p className="text-sm text-green-700 font-semibold">
                            {result.overall_mdr_risk
                              ? "Consider alternative therapy"
                              : "Standard treatment applicable"}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  </div>

                  {/* Resistance Details */}
                  <div className="space-y-4 mt-6">
                    {/* Resistance Summary */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1.0, duration: 0.4 }}
                      className="p-4 rounded-lg border border-yellow-200 bg-gradient-to-br from-yellow-50 to-amber-50"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-yellow-700 font-medium">
                            Resistance Summary
                          </p>
                          <p className="text-yellow-800 font-semibold">
                            {result.total_resistant_count} resistant out of{" "}
                            {result.resistant_antibiotics.length +
                              result.susceptible_antibiotics.length}{" "}
                            tested
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-yellow-700 font-medium">
                            Resistance %
                          </p>
                          <p className="text-yellow-800 text-2xl font-bold">
                            {result.resistance_percentage.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </motion.div>

                    {/* Resistant Antibiotics */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1.1, duration: 0.4 }}
                      className="p-4 rounded-lg border border-red-200 bg-gradient-to-br from-red-50 to-rose-50"
                    >
                      <h4 className="text-lg font-semibold text-red-700 mb-2">
                        Likely to be resistant to {formData.pathogen}
                      </h4>
                      {result.resistant_antibiotics.length > 0 ? (
                        <ul className="list-disc list-inside text-red-800 space-y-1">
                          {result.resistant_antibiotics.map((ab, idx) => (
                            <li key={`resistant-${idx}`}>{ab.name}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-red-600 italic">None found.</p>
                      )}
                    </motion.div>

                    {/* Susceptible Antibiotics */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1.2, duration: 0.4 }}
                      className="p-4 rounded-lg border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50"
                    >
                      <h4 className="text-lg font-semibold text-green-700 mb-2">
                        Likely to be susceptible to {formData.pathogen}
                      </h4>
                      {result.susceptible_antibiotics.length > 0 ? (
                        <ul className="list-disc list-inside text-green-800 space-y-1">
                          {result.susceptible_antibiotics.map((ab, idx) => (
                            <li key={`susceptible-${idx}`}>{ab.name}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-green-600 italic">None found.</p>
                      )}
                    </motion.div>
                  </div>
                </>
              
              </div>
          </AnimatedCard>)
            }
        </motion.div>
      </div>
    </div>
  );
}
