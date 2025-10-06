"use client"

import { useState } from "react"
import { LLMInterface } from "@/components/llm-interface"
import { ConversationHistory } from "@/components/conversation-history"
import { ModelSelector } from "@/components/model-selector"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, Database } from "lucide-react"

export interface Message {
  id: string
  model: string
  userPrompt: string
  response: string
  timestamp: Date
  responseTime: number
  success: boolean
}

export interface LLMModel {
  id: string
  name: string
  endpoint: string
  description: string
  category: "coding" | "conversation"
}

export interface DatabaseRecord {
  id: number
  model_name: string
  system_prompt: string
  user_prompt: string
  response_text: string
  created_at: string
  response_time_ms: number
  success: boolean
}

const models: LLMModel[] = [
  {
    id: "llamasharp",
    name: "LlamaSharp",
    endpoint: "/llamasharp",
    description: "Specialized for C# coding",
    category: "coding",
  },
  {
    id: "codeqwen",
    name: "CodeQwen",
    endpoint: "/codeqwen",
    description: "Python/JavaScript coding",
    category: "coding",
  },
  {
    id: "codellama",
    name: "CodeLlama",
    endpoint: "/codellama",
    description: "Python/JavaScript coding",
    category: "coding",
  },
  {
    id: "mistral",
    name: "Mistral",
    endpoint: "/mistral",
    description: "General conversation",
    category: "conversation",
  },
  {
    id: "phi3",
    name: "Phi-3",
    endpoint: "/phi3",
    description: "General conversation",
    category: "conversation",
  },
  {
    id: "llava",
    name: "LLava",
    endpoint: "/llava",
    description: "Image to Text",
    category: "conversation",
  },
]

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<LLMModel>(models[0])
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [currentResponse, setCurrentResponse] = useState<string>("")

  const handleSearchDatabase = async () => {
    if (!searchTerm.trim()) return

    setIsSearching(true)
    try {
      const response = await fetch(`http://localhost:8000/interaction/${encodeURIComponent(searchTerm)}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const records: DatabaseRecord[] = await response.json()

      const searchMessages: Message[] = records.map((record) => ({
        id: record.id.toString(),
        model: record.model_name,
        userPrompt: record.user_prompt,
        response: record.response_text,
        timestamp: new Date(record.created_at),
        responseTime: record.response_time_ms,
        success: record.success,
      }))

      setMessages(searchMessages)
      setCurrentResponse(`Found ${records.length} matching conversations from database`)
    } catch (error) {
      console.error("Error searching database:", error)
      setCurrentResponse(`Error searching database: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsSearching(false)
    }
  }

  const handleSendMessage = async (prompt: string) => {
    if (!prompt.trim()) return

    setIsLoading(true)
    setCurrentResponse("")
    const startTime = Date.now()

    try {
      const response = await fetch(`http://localhost:8000${selectedModel.endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_prompt: prompt }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const responseTime = Date.now() - startTime

      const newMessage: Message = {
        id: data.prompt_id || Date.now().toString(),
        model: selectedModel.name,
        userPrompt: prompt,
        response: data.response,
        timestamp: new Date(),
        responseTime: data.response_time_ms || responseTime,
        success: true,
      }

      setMessages((prev) => [newMessage, ...prev])
      setCurrentResponse(data.response)
    } catch (error) {
      console.error("Error sending message:", error)
      const errorResponse = `Error: ${error instanceof Error ? error.message : "Unknown error occurred"}`
      const newMessage: Message = {
        id: Date.now().toString(),
        model: selectedModel.name,
        userPrompt: prompt,
        response: errorResponse,
        timestamp: new Date(),
        responseTime: Date.now() - startTime,
        success: false,
      }
      setMessages((prev) => [newMessage, ...prev])
      setCurrentResponse(errorResponse)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearHistory = () => {
    setMessages([])
    setCurrentResponse("")
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold text-card-foreground">Local LLM Interface</h1>
              <ModelSelector models={models} selectedModel={selectedModel} onModelChange={setSelectedModel} />
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <div className="xl:col-span-2">
            <Card className="p-6">
              <LLMInterface selectedModel={selectedModel} onSendMessage={handleSendMessage} isLoading={isLoading} />
            </Card>

            <Card className="p-6 mt-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Search Database History</h3>
                <div className="flex gap-2">
                  <Input
                    placeholder="Search existing conversations..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearchDatabase()}
                    className="flex-1"
                  />
                  <Button onClick={handleSearchDatabase} disabled={!searchTerm.trim() || isSearching}>
                    {isSearching ? (
                      <>
                        <Database className="w-4 h-4 mr-2 animate-pulse" />
                        Searching...
                      </>
                    ) : (
                      <>
                        <Search className="w-4 h-4 mr-2" />
                        Search
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        </div>
        <br />
        <div className="grid grid-flow-col grid-rows-3 gap-4">
          <div className="xl:row-span-3">
            <ConversationHistory messages={messages} onClearHistory={handleClearHistory} />
          </div>
        </div>
        <br />
        {/* <div className="grid grid-flow-col grid-rows-3 gap-4">
          <div className="xl:row-span-3">
            <Card className="p-6 h-[calc(100vh-12rem)]">
              <h3 className="text-lg font-semibold mb-4">Current Response</h3>
              <div className="h-full overflow-auto">
                {currentResponse ? (
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded-md">{currentResponse}</pre>
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground py-8">
                    <p>Response will appear here after sending a message or searching the database.</p>
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div> */}
      </main>
    </div>
  )
}
