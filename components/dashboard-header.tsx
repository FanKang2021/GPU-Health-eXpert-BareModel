"use client"
import { Button } from "@/components/ui/button"
import { Globe, ChevronDown, Monitor } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface DashboardHeaderProps {
  language: "zh" | "en"
  onLanguageToggle: () => void
  t: any
  currentPage: string
  onPageChange: (page: string) => void
}

export function DashboardHeader({ language, onLanguageToggle, t, currentPage, onPageChange }: DashboardHeaderProps) {
  const navItems = {
    zh: [
      { id: "dashboard", label: "主页" },
      { id: "troubleshooting", label: "自检专区" },
      { id: "burn-in", label: "烧机专区" },
    ],
    en: [
      { id: "dashboard", label: "Dashboard" },
      { id: "troubleshooting", label: "Self-Check" },
      { id: "burn-in", label: "Burn-in" },
    ],
  }

  const items = navItems[language]

  return (
    <div className="sticky top-0 z-50 border-b border-border/50 transition-colors duration-200 bg-card/80 backdrop-blur-tech">
      <div className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            {/* Logo and Title with Animation */}
            <div className="flex items-center space-x-3">
              <div className="relative group cursor-pointer">
                {/* Outer rotating ring */}
                <div
                  className="absolute -inset-3 bg-gradient-to-r from-tech-blue/20 via-tech-purple/20 to-tech-cyan/20 rounded-full blur-sm group-hover:blur-md transition-all duration-500 animate-spin"
                  style={{ animationDuration: "8s" }}
                />
                {/* Middle pulse ring */}
                <div className="absolute -inset-2 bg-gradient-to-r from-tech-blue/30 to-tech-purple/30 rounded-full animate-pulse-slow" />
                {/* Logo container */}
                <div className="relative p-2.5 bg-gradient-to-br from-tech-blue/20 to-tech-purple/20 rounded-xl backdrop-blur-sm border border-tech-blue/30 group-hover:scale-110 transition-all duration-300">
                  <Monitor className="w-6 h-6 text-tech-blue animate-glow group-hover:animate-bounce" />
                  {/* Decorative dots */}
                  <div className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-tech-green rounded-full animate-pulse shadow-glow-green" />
                  <div
                    className="absolute -bottom-1 -left-1 w-2 h-2 bg-tech-yellow rounded-full animate-pulse shadow-glow"
                    style={{ animationDelay: "0.5s" }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-500 bg-clip-text text-transparent animate-pulse">
                    GHealthX
                  </h1>
                  <div className="flex space-x-1">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                    <div
                      className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"
                      style={{ animationDelay: "0.3s" }}
                    />
                    <div
                      className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-pulse"
                      style={{ animationDelay: "0.6s" }}
                    />
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <p className="text-xs text-cyan-400 font-mono tracking-wide">GPU Health Expert</p>
                  <div className="flex items-center space-x-1.5 px-2 py-0.5 bg-green-500/20 rounded-full border border-green-500/30">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-xs text-green-500 font-mono">LIVE</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="hidden md:flex items-center space-x-1 ml-8">
              {items.map((item) => (
                <Button
                  key={item.id}
                  variant="ghost"
                  size="sm"
                  onClick={() => onPageChange(item.id)}
                  className={`
                    transition-all duration-300 relative overflow-hidden font-medium
                    ${
                      currentPage === item.id
                        ? "bg-gradient-primary text-primary-foreground shadow-glow"
                        : "text-foreground hover:bg-secondary/50 hover:text-secondary-foreground"
                    }
                  `}
                >
                  {currentPage === item.id && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
                  )}
                  <span className="relative">{item.label}</span>
                </Button>
              ))}
            </div>
          </div>

          {/* Language Switcher */}
          <div className="flex items-center space-x-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="tech-button border-tech-blue/50 hover:bg-tech-blue/20 hover:border-tech-blue bg-transparent"
                >
                  <Globe className="w-4 h-4 mr-2" />
                  <span className="font-mono">{language === "zh" ? "中文" : "English"}</span>
                  <ChevronDown className="w-3 h-3 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="tech-card bg-gradient-to-br from-secondary/20 to-secondary/10 border-border/50 shadow-glow backdrop-blur-tech">
                <DropdownMenuItem
                  onClick={() => language !== "zh" && onLanguageToggle()}
                  className={`cursor-pointer transition-all duration-300 ${
                    language === "zh"
                      ? "bg-tech-blue/20 text-tech-blue font-semibold"
                      : "text-foreground hover:bg-tech-blue/10 hover:text-tech-blue"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">🇨🇳</span>
                    <span className="font-mono">中文</span>
                    {language === "zh" && <span className="text-tech-blue">✓</span>}
                  </div>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => language !== "en" && onLanguageToggle()}
                  className={`cursor-pointer transition-all duration-300 ${
                    language === "en"
                      ? "bg-tech-blue/20 text-tech-blue font-semibold"
                      : "text-foreground hover:bg-tech-blue/10 hover:text-tech-blue"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">🇺🇸</span>
                    <span className="font-mono">English</span>
                    {language === "en" && <span className="text-tech-blue">✓</span>}
                  </div>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </div>
  )
}
